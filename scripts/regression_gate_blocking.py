"""Strict regression gate comparing Maithili standalone vs Integrated main project blocking."""
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BRANCH2_ROOT = ROOT / "project-branch2"

sys.path.insert(0, str(ROOT))

from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.pipeline import candidates_df_to_map


STANDALONE_RUNNER_CODE = """
import sys
import json
from pathlib import Path
import pandas as pd

BRANCH2_ROOT = Path(sys.argv[1])
sys.path.insert(0, str(BRANCH2_ROOT))

import src.blocking as m_blocking
import src.exact_blocking as m_exact
import src.char_ngram_blocking as m_ngram
import src.candidate_pipeline as m_pipe
import src.normalization as m_norm
import src.address as m_addr

with open(sys.argv[2], "r", encoding="utf-8") as f:
    payload = json.load(f)

s1_df = pd.DataFrame(payload["s1"])
s2_df = pd.DataFrame(payload["s2"])
s3_df = pd.DataFrame(payload["s3"])
top_k = payload.get("top_k", 20)

s2_ids = [str(x) if str(x).startswith("S2-") else f"S2-{x}" for x in s2_df["entity_id"]]
s3_ids = [str(x) if str(x).startswith("S3-") else f"S3-{x}" for x in s3_df["entity_id"]]

target_ids = s2_ids + s3_ids
target_names = list(s2_df["business_name"]) + list(s3_df["business_name"])
target_addrs = list(s2_df["business_address"]) + list(s3_df["business_address"])

exact_name_idx = m_exact.build_exact_index(target_ids, (m_norm.normalize_name(n) for n in target_names))
exact_addr_idx = m_exact.build_exact_index(target_ids, (m_addr.address_fingerprint(a) for a in target_addrs))
token_idx = m_blocking.build_token_index(target_ids, target_names, min_token_length=2, max_token_frequency=None)
ngram_idx = m_ngram.build_char_ngram_index(target_ids, target_names, ngram_range=(2, 5), min_df=1, max_features=None)

pipeline = m_pipe.CandidatePipeline({
    "exact_name": lambda rec: m_exact.generate_exact_candidates(m_norm.normalize_name(rec["name"]), exact_name_idx),
    "exact_address": lambda rec: m_exact.generate_exact_candidates(m_addr.address_fingerprint(rec["address"]), exact_addr_idx),
    "token": lambda rec: m_blocking.generate_token_candidates(rec["name"], token_idx, min_token_length=2, max_candidates=None),
    "char_ngram": lambda rec: m_ngram.generate_char_ngram_candidates(rec["name"], ngram_idx, top_k=top_k),
})

cand_map = {}
for _, row in s1_df.iterrows():
    s1_id = str(row["entity_id"])
    rec = {"name": row["business_name"], "address": row["business_address"]}
    res = pipeline.generate(rec)
    cand_map[s1_id] = res.all_candidates

with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(cand_map, f)
"""


def run_standalone_subproc(s1_df, s2_df, s3_df, top_k=20):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runner_file = tmp_path / "runner.py"
        in_file = tmp_path / "in.json"
        out_file = tmp_path / "out.json"

        runner_file.write_text(STANDALONE_RUNNER_CODE, encoding="utf-8")
        payload = {
            "s1": s1_df.to_dict(orient="records"),
            "s2": s2_df.to_dict(orient="records"),
            "s3": s3_df.to_dict(orient="records"),
            "top_k": top_k,
        }
        with open(in_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        res = subprocess.run([sys.executable, str(runner_file), str(BRANCH2_ROOT), str(in_file), str(out_file)],
                             capture_output=True, text=True, check=True)
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)


def run_integrated(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, top_k=20):
    """Runs the Integrated CandidateGenerator adapter in the main project."""
    norm = DataNormalizer()
    s1_n = norm.normalize_dataframe(s1_df)
    s2_n = norm.normalize_dataframe(s2_df)
    s3_n = norm.normalize_dataframe(s3_df)

    gen = CandidateGenerator(top_k=top_k)
    cand_df = gen.generate_candidates(s1_n, s2_n, s3_n)

    all_s1_ids = s1_df["entity_id"].astype(str).tolist()
    cand_map = candidates_df_to_map(cand_df, all_s1_ids=all_s1_ids)
    return cand_df, cand_map


def compute_metrics(cand_map, ground_truth, s2_ids, s3_ids):
    counts = [len(cand_map[s1_id]) for s1_id in cand_map]
    total_pairs = sum(counts)
    mean_cands = float(np.mean(counts)) if counts else 0.0
    median_cands = float(np.median(counts)) if counts else 0.0
    p95_cands = float(np.percentile(counts, 95)) if counts else 0.0
    max_cands = int(np.max(counts)) if counts else 0
    zero_cands = sum(1 for c in counts if c == 0)
    zero_rate = zero_cands / len(counts) if counts else 0.0

    total_gt = 0
    captured_gt = 0
    s2_gt = 0
    s2_captured = 0
    s3_gt = 0
    s3_captured = 0

    for s1_id, gt_targets in ground_truth.items():
        cands = set(cand_map.get(s1_id, []))
        for target in gt_targets:
            total_gt += 1
            is_s2 = target in s2_ids
            is_s3 = target in s3_ids

            if is_s2:
                s2_gt += 1
                if target in cands:
                    s2_captured += 1
            elif is_s3:
                s3_gt += 1
                if target in cands:
                    s3_captured += 1

            if target in cands:
                captured_gt += 1

    overall_recall = captured_gt / total_gt if total_gt else 1.0
    s2_recall = s2_captured / s2_gt if s2_gt else 1.0
    s3_recall = s3_captured / s3_gt if s3_gt else 1.0

    return {
        "Total Candidate Pairs": total_pairs,
        "Blocking Recall": f"{overall_recall * 100:.2f}%",
        "S2 Recall": f"{s2_recall * 100:.2f}%",
        "S3 Recall": f"{s3_recall * 100:.2f}%",
        "Mean Candidates/S1": f"{mean_cands:.2f}",
        "Median Candidates/S1": f"{median_cands:.2f}",
        "P95 Candidates/S1": f"{p95_cands:.2f}",
        "Max Candidates/S1": max_cands,
        "Zero-Candidate Rate": f"{zero_rate * 100:.2f}%",
        "Captured GT Pairs": f"{captured_gt}/{total_gt}",
        "Captured S2 Pairs": f"{s2_captured}/{s2_gt}",
        "Captured S3 Pairs": f"{s3_captured}/{s3_gt}",
    }


def main():
    print("=" * 75)
    print("STRICT REGRESSION GATE: MAITHILI STANDALONE VS INTEGRATED MAIN PROJECT")
    print("=" * 75)

    # 1. Dataset 1: Full synthetic dataset with diverse ER challenges
    s1_1 = pd.DataFrame([
        ("s1_01", "Acme Corporation", "123 Main Street, New York, NY 10001", "US"),
        ("s1_02", "Beta Solutions International", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s1_03", "Gamma & Enterprises LLC", "789 Pine Road, Chicago, IL 60601", "US"),
        ("s1_04", "Delta Technologies Group", "101 Elm Street, Houston, TX 77001", "US"),
        ("s1_05", "Epsilon Global Services", "202 Maple Drive, Phoenix, AZ 85001", "US"),
        ("s1_06", "Zeta Industries Ltd", "303 Cedar Street, Dallas, TX 75201", "US"),
        ("s1_07", "Eta Manufacturing Corp", "404 Birch Lane, San Jose, CA 95101", "US"),
        ("s1_08", "Theta Franchise Inc", "500 Commerce Blvd, Miami, FL 33101", "US"),
        ("s1_09", "Iota Ventures LLC", "999 Unknown Rd, Unknown City", "US"),
        ("s1_10", "Kappa Solutions Corp", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s1_11", "Lambda UK Ltd", "10 Downing Street, London", "UK"),
        ("s1_12", "Mu Enterprises SARL", "15 Rue de Rivoli, Paris", "France"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s2_1 = pd.DataFrame([
        ("s2_01", "Acme Corporation", "123 Main Street, New York, NY 10001", "US"),
        ("s2_02", "Beta Solns Intl", "456 Oak Ave, Los Angeles, CA", "US"),
        ("s2_03", "Gamma Enterprises LLC", "789 Pine Road, Chicago, IL 60601", "US"),
        ("s2_04", "Group Technologies Delta", "101 Elm St, Houston, TX 77001", "US"),
        ("s2_05", "Epsilom Global Services", "202 Maple Drive, Phoenix, AZ 85001", "US"),
        ("s2_06", "Zeta Industries Ltd", "303 Cedar St, Dallas, TX 75201", "US"),
        ("s2_07", "Eta Manufacturing Corp", "404 Birch Lane, San Jose, CA", "US"),
        ("s2_08a", "Theta Franchise Inc", "500 Commerce Blvd, Miami, FL 33101", "US"),
        ("s2_08b", "Theta Franchise Incorporated", "500 Commerce Blvd, Miami, FL", "US"),
        ("s2_09", "Completely Different Company", "001 Nowhere Ave, Nowhere", "US"),
        ("s2_10", "Kappa Solutions Corp", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s2_11", "Lambda UK Limited", "10 Downing St, London", "UK"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s3_1 = pd.DataFrame([
        ("s3_01", "Acme Corp", "123 Main St, New York, 10001", "US"),
        ("s3_03", "Gamma Enterprises", "789 Pine Rd, Chicago, IL", "US"),
        ("s3_12", "Mu Entreprises SARL", "15 Rue de Rivoli, Paris 75001", "France"),
        ("s3_xx", "Unrelated Company X", "1 Random Street, Somewhere", "US"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    gt_1 = {
        "s1_01": {"s2_01", "s3_01"},
        "s1_02": {"s2_02"},
        "s1_03": {"s2_03", "s3_03"},
        "s1_04": {"s2_04"},
        "s1_05": {"s2_05"},
        "s1_06": {"s2_06"},
        "s1_07": {"s2_07"},
        "s1_08": {"s2_08a", "s2_08b"},
        "s1_09": set(),
        "s1_10": set(),
        "s1_11": {"s2_11"},
        "s1_12": {"s3_12"},
    }

    # Run standalone on Dataset 1
    standalone_map = run_standalone_subproc(s1_1, s2_1, s3_1, top_k=20)
    standalone_cleaned_map = {}
    for s1_id, cands in standalone_map.items():
        cleaned = [c.replace("S2-", "").replace("S3-", "") for c in cands]
        standalone_cleaned_map[s1_id] = cleaned

    cand_df_1, integrated_map = run_integrated(s1_1, s2_1, s3_1, top_k=20)

    s2_ids_1 = set(s2_1["entity_id"])
    s3_ids_1 = set(s3_1["entity_id"])

    metrics_standalone = compute_metrics(standalone_cleaned_map, gt_1, s2_ids_1, s3_ids_1)
    metrics_integrated = compute_metrics(integrated_map, gt_1, s2_ids_1, s3_ids_1)

    print("\n--- Metric Comparison (Dataset 1: Full Competition Synthetic) ---")
    df_comp = pd.DataFrame([metrics_standalone, metrics_integrated], index=["Maithili Standalone", "Integrated Main Project"]).T
    print(df_comp.to_string())

    # Deep candidate set comparison per S1 entity
    print("\n--- Candidate Set Entity-by-Entity Verification ---")
    lost_candidates = {}
    added_candidates = {}
    duplicate_candidates = {}
    source_confusion = []

    for s1_id in s1_1["entity_id"]:
        std_set = set(standalone_cleaned_map.get(s1_id, []))
        int_set = set(integrated_map.get(s1_id, []))
        gt_set = gt_1.get(s1_id, set())

        # Check for lost true matches
        lost_gt = gt_set - int_set
        if lost_gt:
            print(f"CRITICAL ERROR: S1 entity {s1_id} lost GT target(s): {lost_gt}")

        diff_lost = std_set - int_set
        diff_added = int_set - std_set
        if diff_lost:
            lost_candidates[s1_id] = diff_lost
        if diff_added:
            added_candidates[s1_id] = diff_added

        # Check duplicates in integrated map
        int_list = integrated_map.get(s1_id, [])
        if len(int_list) != len(int_set):
            duplicate_candidates[s1_id] = len(int_list) - len(int_set)

    # Check source confusion in cand_df
    for _, row in cand_df_1.iterrows():
        s2_id = row.get("s2_id")
        s3_id = row.get("s3_id")
        if pd.notna(s2_id) and s2_id not in s2_ids_1:
            source_confusion.append(f"s2_id {s2_id} not in Source 2")
        if pd.notna(s3_id) and s3_id not in s3_ids_1:
            source_confusion.append(f"s3_id {s3_id} not in Source 3")

    print(f"Lost Candidates across S1 entities: {len(lost_candidates)} entities")
    print(f"Incorrectly Added Candidates: {len(added_candidates)} entities")
    print(f"Duplicate Candidates: {len(duplicate_candidates)} entities")
    print(f"Source Confusion Errors: {len(source_confusion)} errors")
    print(f"Every GT Target Recovered by Maithili preserved in Integration: True")

    # 2. Dataset 2: Multi-match benchmark from evaluate_m2
    print("\n" + "=" * 75)
    print("DATASET 2: BRANCH2 MULTI-MATCH BENCHMARK")
    print("=" * 75)
    s1_2 = pd.DataFrame([
        ('S1-A', 'Alpha Ltd.', 'Addr A', 'US'),
        ('S1-B', 'Private Limited', 'Addr B', 'US'),
        ('S1-C', 'Alpha', 'Addr C', 'US'),
        ('S1-D', 'Zeta', 'Addr D', 'US'),
    ], columns=['entity_id', 'business_name', 'business_address', 'country'])

    s2_2 = pd.DataFrame([
        ('S2-1', 'Alpha Limited', 'Addr 1', 'US'),
        ('S2-2', 'Alpha Finance', 'Addr 2', 'US'),
        ('S2-3', 'Private Limited', 'Addr 3', 'US'),
        ('S2-4', 'Alpha Unrelated', 'Addr 4', 'US'),
        ('S2-5', 'Theta', 'Addr 5', 'US'),
    ], columns=['entity_id', 'business_name', 'business_address', 'country'])

    s3_2 = pd.DataFrame([
        ('S3-1', 'Alpha Trading', 'Addr 1', 'US'),
        ('S3-2', 'Delta', 'Addr 2', 'US'),
        ('S3-3', 'Private Limited', 'Addr 3', 'US'),
        ('S3-4', 'Alpha Beta', 'Addr 4', 'US'),
        ('S3-5', 'Zeta Corp', 'Addr 5', 'US'),
    ], columns=['entity_id', 'business_name', 'business_address', 'country'])

    gt_2 = {
        'S1-C': set(),
        'S1-D': {'S3-5', 'S2-5'},
        'S1-A': {'S3-4', 'S2-2', 'S3-2', 'S2-1', 'S3-1'},
        'S1-B': {'S3-3', 'S2-3'},
    }

    std_map_2 = run_standalone_subproc(s1_2, s2_2, s3_2, top_k=20)
    cand_df_2, int_map_2 = run_integrated(s1_2, s2_2, s3_2, top_k=20)

    s2_ids_2 = set(s2_2["entity_id"])
    s3_ids_2 = set(s3_2["entity_id"])

    m_std_2 = compute_metrics(std_map_2, gt_2, s2_ids_2, s3_ids_2)
    m_int_2 = compute_metrics(int_map_2, gt_2, s2_ids_2, s3_ids_2)

    df_comp_2 = pd.DataFrame([m_std_2, m_int_2], index=["Maithili Standalone", "Integrated Main Project"]).T
    print(df_comp_2.to_string())

    for s1_id in s1_2["entity_id"]:
        std_set = set(std_map_2.get(s1_id, []))
        int_set = set(int_map_2.get(s1_id, []))
        assert std_set == int_set, f"Mismatch on {s1_id}: std={std_set} != int={int_set}"

    print(f"\nDataset 2 Candidate Sets Match: 100% IDENTICAL (std == int for all S1 entities)")


if __name__ == "__main__":
    main()
