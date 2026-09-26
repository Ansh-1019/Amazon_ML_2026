#!/usr/bin/env python3
"""Evaluate raw M2 name blocking without materializing a candidate-pair table.

Run from any directory with Python 3.10+ and the repository requirements:
    python /path/to/repo/scripts/evaluate_m2.py --data-dir /path/to/train --include-m1

All S1 rows contribute candidate volume/distributions, including unmatched rows.
Ground truth contributes unique labeled pairs only. Source and ground-truth S1
IDs must be unique and cover the same records; invalid/dangling labels raise errors.
Missing names are empty strings, never the literal string 'nan'. Literal name
values such as 'NA' are preserved. Missing matched_entity_ids mean zero matches.

Ground-truth cells contain comma-separated IDs. Malformed IDs raise an error
rather than silently reporting incorrect recall. Duplicate IDs in a cell count
once.

M2 always uses default stopwords, min_token_length=2, no frequency pruning and
no candidate cap. Optional M1 uses literal normalize_name equality, including
empty normalized names. The union deduplicates pairs independently per source.

Memory: loaded ID/name columns, aligned raw ground truth, inverted indexes,
optional exact-name indexes, 2 uint64 counts per S1, and one transient M2 list.
No complete candidate lists persist between queries. Unpruned indexes and a
single query's candidate list can still be large; this run measures that cost.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from pathlib import Path
import re
import sys
from time import perf_counter

# Support direct script execution outside the repository working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.blocking import build_token_index, generate_token_candidates
from src.normalization import normalize_name

ID_PATTERN = re.compile(r'S[23]-[^\s,;|\[\]\'\"]+')
PERCENTILES = (50, 90, 95, 99, 99.9)


@dataclass
class TrainingData:
    sources: dict[str, pd.DataFrame]
    truth: pd.Series  # Raw cells aligned to S1 index, not expanded pair records.
    ground_truth_rows: int


def parse_matches(value: object) -> dict[str, set[str]]:
    """Parse one ground-truth cell and partition unique IDs by source prefix."""
    result: dict[str, set[str]] = {'S2': set(), 'S3': set()}
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return result
    if not isinstance(value, str):
        raise ValueError(f'Expected a string ground-truth cell, got {value!r}')
    value = value.strip()
    if not value or value.lower() in {'nan', 'null', 'none', '<na>'}:
        return result
    ids = [entity_id.strip() for entity_id in value.split(',')]
    for entity_id in ids:
        if not isinstance(entity_id, str) or not ID_PATTERN.fullmatch(entity_id):
            raise ValueError(f'Invalid target ID {entity_id!r} in {value!r}')
        result[entity_id[:2]].add(entity_id)
    return result


def load_data(data_dir: Path) -> TrainingData:
    """Read only ID/name and ground-truth columns, preserving strings."""
    sources = {}
    for source in ('S1', 'S2', 'S3'):
        frame = pd.read_csv(
            data_dir / f'train_source{source[1]}.tsv', sep='\t', dtype=str,
            usecols=['entity_id', 'business_name'], keep_default_na=False,
        ).set_index('entity_id')
        if not frame.index.is_unique:
            raise ValueError(f'{source} contains duplicate entity IDs')
        if any(not entity_id.startswith(source + '-') for entity_id in frame.index):
            raise ValueError(f'{source} contains missing or incorrectly prefixed entity IDs')
        sources[source] = frame
    ground_truth = pd.read_csv(
        data_dir / 'train_ground_truth.tsv', sep='\t', dtype=str,
        usecols=['source1_entity_id', 'matched_entity_ids'], keep_default_na=False,
    ).set_index('source1_entity_id')
    if not ground_truth.index.is_unique:
        raise ValueError('Ground truth contains duplicate S1 IDs')
    s1_ids = sources['S1'].index
    if (len(ground_truth) != len(s1_ids)
            or not ground_truth.index.isin(s1_ids).all()):
        raise ValueError('Ground truth must contain exactly one row for every S1 ID')
    truth = ground_truth['matched_entity_ids'].reindex(s1_ids)
    return TrainingData(sources, truth, len(ground_truth))


def _contains(sorted_ids: list[str], entity_id: str) -> bool:
    position = bisect_left(sorted_ids, entity_id)
    return position < len(sorted_ids) and sorted_ids[position] == entity_id


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(counts: np.ndarray) -> dict[str, float | int]:
    if not len(counts):
        return {**{f'{p}%': 0.0 for p in PERCENTILES}, 'maximum': 0}
    values = np.percentile(counts, PERCENTILES, method='linear')
    return {**{f'{p}%': float(v) for p, v in zip(PERCENTILES, values)},
            'maximum': int(counts.max())}


def _informative_tokens(name: object) -> list[str]:
    # Diagnostics only: use the public blocking API, not a second token filter.
    return list(build_token_index(['diagnostic'], [name]))


def evaluate(
    data: TrainingData, *, include_m1: bool = False,
    missed_examples: int = 20, progress_every: int = 10000,
) -> dict:
    """Evaluate loaded data; candidate pairs are counted, never accumulated."""
    if missed_examples < 0 or progress_every < 0:
        raise ValueError('Diagnostic and progress limits must be non-negative')
    start = perf_counter()
    timings = {}
    indexes = {}
    token_stats = {}
    exact_indexes = {}
    for source in ('S2', 'S3'):
        frame = data.sources[source]
        print(f'Building {source} unpruned M2 index ({len(frame):,} rows)...',
              file=sys.stderr, flush=True)
        tick = perf_counter()
        index = build_token_index(
            frame.index, frame['business_name'], min_token_length=2,
            max_token_frequency=None,
        )
        timings[f'{source} index construction'] = perf_counter() - tick
        indexes[source] = index
        token_stats[source] = {
            'indexed_tokens': len(index),
            'top_tokens': [(token, len(ids)) for token, ids in nsmallest(
                30, index.items(), key=lambda item: (-len(item[1]), item[0]))],
        }
        if include_m1:
            tick = perf_counter()
            exact = defaultdict(list)
            for entity_id, name in frame['business_name'].items():
                exact[normalize_name(name)].append(entity_id)
            exact_indexes[source] = dict(exact)
            del exact
            timings[f'{source} M1 index construction'] = perf_counter() - tick

    totals = {source: {'true_pairs': 0, 'captured_true_pairs': 0,
                       'candidate_pairs': 0} for source in ('S2', 'S3')}
    comparison = {method: {'captured_true_pairs': 0, 'candidate_pairs': 0}
                  for method in ('M1', 'M1 union M2')} if include_m1 else None
    s1 = data.sources['S1']
    counts = np.zeros((len(s1), 2), dtype=np.uint64)
    examples = []
    rows_with_matches = 0
    m1_seconds = 0.0
    tick = perf_counter()
    records = zip(s1.index, s1['business_name'], data.truth, strict=True)
    for row_number, (s1_id, name, raw_matches) in enumerate(records):
        try:
            matches = parse_matches(raw_matches)
        except ValueError as exc:
            raise ValueError(f'Ground truth for {s1_id}: {exc}') from exc
        rows_with_matches += bool(matches['S2'] or matches['S3'])
        normalized = None
        if include_m1:
            m1_tick = perf_counter()
            normalized = normalize_name(name)
            m1_seconds += perf_counter() - m1_tick
        for column, source in enumerate(('S2', 'S3')):
            true_ids = matches[source]
            target = data.sources[source]
            for target_id in true_ids:
                if target_id not in target.index:
                    raise ValueError(f'{s1_id}: true target {target_id} is absent from {source}')
            candidates = generate_token_candidates(
                name, indexes[source], min_token_length=2, max_candidates=None,
            )
            captured = {target_id for target_id in true_ids if _contains(candidates, target_id)}
            counts[row_number, column] = len(candidates)
            totals[source]['true_pairs'] += len(true_ids)
            totals[source]['captured_true_pairs'] += len(captured)
            totals[source]['candidate_pairs'] += len(candidates)
            if include_m1:
                m1_tick = perf_counter()
                exact_candidates = exact_indexes[source].get(normalized, ())
                # Exact lists can be large; only retain the small true-ID subset.
                exact_captured = true_ids.intersection(exact_candidates)
                additional = sum(not _contains(candidates, entity_id)
                                 for entity_id in exact_candidates)
                comparison['M1']['candidate_pairs'] += len(exact_candidates)
                comparison['M1']['captured_true_pairs'] += len(exact_captured)
                comparison['M1 union M2']['candidate_pairs'] += len(candidates) + additional
                comparison['M1 union M2']['captured_true_pairs'] += len(captured | exact_captured)
                m1_seconds += perf_counter() - m1_tick
            if len(examples) < missed_examples:
                for target_id in sorted(true_ids - captured):
                    target_name = target.at[target_id, 'business_name']
                    query_tokens = _informative_tokens(name)
                    target_tokens = _informative_tokens(target_name)
                    examples.append({
                        'S1 ID': s1_id, 'S1 business name': name,
                        'true target ID': target_id, 'true target business name': target_name,
                        'normalized S1 name': normalize_name(name),
                        'normalized target name': normalize_name(target_name),
                        'S1 informative tokens': query_tokens,
                        'target informative tokens': target_tokens,
                        'shared informative tokens': sorted(set(query_tokens) & set(target_tokens)),
                    })
                    if len(examples) >= missed_examples:
                        break
            del candidates
        if progress_every and (row_number + 1) % progress_every == 0:
            print(f'Evaluated {row_number + 1:,}/{len(s1):,} S1 rows '
                  f'({perf_counter() - tick:,.1f}s)', file=sys.stderr, flush=True)
    timings['candidate evaluation (includes optional M1/union and diagnostics)'] = perf_counter() - tick
    if include_m1:
        timings['M1/union evaluation (subset of candidate evaluation)'] = m1_seconds
    for metrics in totals.values():
        metrics['missed_true_pairs'] = metrics['true_pairs'] - metrics['captured_true_pairs']
        metrics['pair_recall'] = _ratio(metrics['captured_true_pairs'], metrics['true_pairs'])
    combined = {key: sum(metrics[key] for metrics in totals.values()) for key in
                ('true_pairs', 'captured_true_pairs', 'missed_true_pairs', 'candidate_pairs')}
    combined['pair_recall'] = _ratio(combined['captured_true_pairs'], combined['true_pairs'])
    naive = len(s1) * (len(data.sources['S2']) + len(data.sources['S3']))
    combined['naive_pair_space'] = naive
    combined['candidate_reduction'] = 1 - combined['candidate_pairs'] / naive if naive else None
    if comparison is not None:
        comparison['M2'] = {key: combined[key] for key in ('captured_true_pairs', 'candidate_pairs')}
        for metrics in comparison.values():
            metrics['pair_recall'] = _ratio(metrics['captured_true_pairs'], combined['true_pairs'])
    distributions = {
        'S1 -> S2': _distribution(counts[:, 0]),
        'S1 -> S3': _distribution(counts[:, 1]),
        'Total per S1': _distribution(counts.sum(axis=1)),
    }
    timings['evaluation including index construction and summaries'] = perf_counter() - start
    return {
        'rows': {source: len(frame) for source, frame in data.sources.items()},
        'ground_truth_rows': data.ground_truth_rows, 'rows_with_matches': rows_with_matches,
        'rows_without_matches': len(s1) - rows_with_matches,
        'tokens': token_stats, 'sources': totals, 'm2': combined,
        'comparison': comparison, 'distributions': distributions,
        'timings': timings, 'missed_examples': examples,
    }


def _percent(value: float | None) -> str:
    return 'N/A (zero denominator)' if value is None else f'{100 * value:.6f}%'


def print_report(report: dict) -> None:
    print('\n' + '=' * 60 + '\nM2 FULL-DATA EVALUATION\n' + '=' * 60)
    print('Configuration: default stopwords; min_token_length=2; '
          'max_token_frequency=None; max_candidates=None')
    print('Candidate scope: ALL S1 rows; percentiles: linear interpolation.')
    for source, rows in report['rows'].items():
        print(f'{source} rows: {rows:,}')
    print(f"Ground-truth rows: {report['ground_truth_rows']:,}")
    print(f"S1 rows with / without matches: {report['rows_with_matches']:,} / "
          f"{report['rows_without_matches']:,}")
    for source, stats in report['tokens'].items():
        print(f"{source} indexed tokens: {stats['indexed_tokens']:,}")
    print('\nIndex construction:')
    for source in ('S2', 'S3'):
        print(f"  {source}: {report['timings'][source + ' index construction']:,.2f}s")
    print('\nM2 CANDIDATES')
    for source, metrics in report['sources'].items():
        print(f"  S1 -> {source}: {metrics['candidate_pairs']:,}")
    m2 = report['m2']
    print(f"  TOTAL: {m2['candidate_pairs']:,}")
    print(f"  Naive pair space: {m2['naive_pair_space']:,}")
    print(f"  Candidate reduction: {_percent(m2['candidate_reduction'])}")
    print('\nM2 RECALL')
    for label, key in (('True pairs', 'true_pairs'), ('Captured', 'captured_true_pairs'),
                       ('Missed', 'missed_true_pairs')):
        print(f'  {label}: {m2[key]:,}')
    print(f"  Pair recall: {_percent(m2['pair_recall'])}")
    for source, metrics in report['sources'].items():
        print(f"  {source}: true={metrics['true_pairs']:,}; "
              f"captured={metrics['captured_true_pairs']:,}; "
              f"recall={_percent(metrics['pair_recall'])}")
    print('\nCANDIDATE DISTRIBUTION')
    for source, distribution in report['distributions'].items():
        print(f"  {source}: " + '; '.join(f'{key}={value:,.2f}' for key, value in distribution.items()))
    print('\nM1 vs M2 vs M1 UNION M2')
    if report['comparison'] is None:
        print('  Disabled; enable with --include-m1.')
    else:
        print('  M1 uses normalized-name equality, including empty names.')
        for method in ('M1', 'M2', 'M1 union M2'):
            metrics = report['comparison'][method]
            print(f"  {method}: recall={_percent(metrics['pair_recall'])}; "
                  f"candidates={metrics['candidate_pairs']:,}")
    print('\nTIMING (overlapping subtotals are labeled; do not add them)')
    for stage, seconds in report['timings'].items():
        print(f'  {stage}: {seconds:,.2f}s')
    print('\nTOP TOKEN POSTING LISTS (up to 30 per source; ties alphabetical)')
    for source, stats in report['tokens'].items():
        print(f'  {source}:')
        for token, size in stats['top_tokens']:
            print(f'    {token!r} -> {size:,}')
    print('\nMISSED-PAIR EXAMPLES (first in S1 file order, S2 then S3, target ID sorted)')
    if not report['missed_examples']:
        print('  No examples collected.')
    for number, example in enumerate(report['missed_examples'], 1):
        print(f'  Example {number}:')
        for label, value in example.items():
            print(f'    {label}: {value!r}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing the training TSV files')
    parser.add_argument('--include-m1', action='store_true', help='Also evaluate exact names and M1 union M2')
    parser.add_argument('--missed-examples', type=int, default=20)
    parser.add_argument('--progress-every', type=int, default=10000, help='S1 progress interval; 0 disables')
    args = parser.parse_args(argv)
    if args.missed_examples < 0 or args.progress_every < 0:
        parser.error('--missed-examples and --progress-every must be non-negative')
    total_start = perf_counter()
    print(f'Loading ID/name columns from {args.data_dir}...', file=sys.stderr, flush=True)
    data = load_data(args.data_dir)
    loading_seconds = perf_counter() - total_start
    report = evaluate(data, include_m1=args.include_m1,
                      missed_examples=args.missed_examples, progress_every=args.progress_every)
    report['timings'] = {'loading': loading_seconds, **report['timings'],
                         'total (before report printing)': perf_counter() - total_start}
    print_report(report)


if __name__ == '__main__':
    main()
