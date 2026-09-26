"""Small fixtures only: never load the full training dataset in tests."""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.evaluate_m2 import evaluate, load_data, parse_matches, print_report


@pytest.fixture
def training_dir(tmp_path):
    rows = {
        '1': [('S1-1', 'Alpha Ltd.'), ('S1-2', 'Private Limited'),
              ('S1-3', 'Alpha'), ('S1-4', ''), ('S1-5', 'Omega')],
        '2': [('S2-1', 'Alpha Limited'), ('S2-2', 'Private Limited'),
              ('S2-3', 'Alpha Beta')],
        '3': [('S3-1', 'Alpha Finance'), ('S3-2', ''), ('S3-3', 'Gamma')],
    }
    for number, records in rows.items():
        frame = pd.DataFrame(records, columns=['entity_id', 'business_name'])
        frame['business_address'] = 'unused address'
        frame['country'] = 'NA'
        frame.to_csv(tmp_path / f'train_source{number}.tsv', sep='\t', index=False)
    # Different order from S1; duplicate labels must not duplicate true pairs.
    pd.DataFrame([
        ('S1-5', None), ('S1-2', 'S2-2'), ('S1-1', 'S2-1,S3-1,S2-1'),
        ('S1-4', 'S3-2'), ('S1-3', None),
    ], columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        tmp_path / 'train_ground_truth.tsv', sep='\t', index=False,
    )
    return tmp_path


@pytest.mark.parametrize('value', [None, float('nan'), pd.NA, '', '  ', 'NaN'])
def test_missing_truth(value):
    assert parse_matches(value) == {'S2': set(), 'S3': set()}


def test_comma_parser_separates_sources_and_deduplicates():
    assert parse_matches('S2-681193310, S2-743505751,S3-775321672,S2-681193310') == {
        'S2': {'S2-681193310', 'S2-743505751'}, 'S3': {'S3-775321672'},
    }


@pytest.mark.parametrize('value', ['S1-1', 'S4-1', 'S2-1 S3-1', 'S2-1,', 123])
def test_bad_truth_fails_explicitly(value):
    with pytest.raises(ValueError):
        parse_matches(value)


def test_hand_calculated_metrics_union_and_diagnostics(training_dir, capsys):
    data = load_data(training_dir)
    assert list(data.sources['S1'].columns) == ['business_name']
    assert data.sources['S1'].at['S1-4', 'business_name'] == ''
    assert data.truth.loc['S1-1'] == 'S2-1,S3-1,S2-1'
    report = evaluate(data, include_m1=True, progress_every=0)
    assert report['rows_with_matches'] == 3
    assert report['rows_without_matches'] == 2
    assert report['m2'] == {
        'true_pairs': 4, 'captured_true_pairs': 2, 'missed_true_pairs': 2,
        'candidate_pairs': 6, 'pair_recall': .5, 'naive_pair_space': 30,
        'candidate_reduction': .8,
    }
    for source, volume in [('S2', 4), ('S3', 2)]:
        assert report['sources'][source] == {
            'true_pairs': 2, 'captured_true_pairs': 1, 'missed_true_pairs': 1,
            'candidate_pairs': volume, 'pair_recall': .5,
        }
    assert report['comparison']['M1'] == {
        'captured_true_pairs': 3, 'candidate_pairs': 3, 'pair_recall': .75,
    }
    assert report['comparison']['M1 union M2'] == {
        'captured_true_pairs': 4, 'candidate_pairs': 8, 'pair_recall': 1.,
    }
    assert report['comparison']['M2']['candidate_pairs'] == 6
    assert report['distributions']['Total per S1'] == {
        '50%': 0., '90%': 3., '95%': 3., '99%': 3., '99.9%': 3., 'maximum': 3,
    }
    assert report['distributions']['S1 -> S2']['maximum'] == 2
    assert report['distributions']['S1 -> S3']['maximum'] == 1
    assert report['tokens']['S2']['top_tokens'][0] == ('alpha', 2)
    examples = report['missed_examples']
    assert [example['true target ID'] for example in examples] == ['S2-2', 'S3-2']
    assert examples[0]['S1 business name'] == 'Private Limited'
    assert examples[0]['shared informative tokens'] == []
    assert examples[1]['normalized target name'] == ''
    print_report(report)
    printed = capsys.readouterr().out
    for section in ('M2 FULL-DATA EVALUATION', 'M2 CANDIDATES', 'M2 RECALL',
                    'CANDIDATE DISTRIBUTION', 'M1 vs M2 vs M1 UNION M2',
                    'TIMING', 'TOP TOKEN POSTING LISTS', 'MISSED-PAIR EXAMPLES'):
        assert section in printed


def test_optional_m1_and_example_limit(training_dir):
    report = evaluate(load_data(training_dir), missed_examples=1, progress_every=0)
    assert report['comparison'] is None
    assert report['m2']['candidate_pairs'] == 6
    assert len(report['missed_examples']) == 1
    assert report['missed_examples'][0]['true target ID'] == 'S2-2'


def test_no_true_pairs_and_literal_na_name(training_dir):
    data = load_data(training_dir)
    data.truth[:] = ''
    report = evaluate(data, missed_examples=0, progress_every=0)
    assert report['m2']['true_pairs'] == 0
    assert report['m2']['pair_recall'] is None
    assert report['m2']['candidate_pairs'] == 6
    file = training_dir / 'train_source1.tsv'
    file.write_text(file.read_text().replace('Omega', 'NA'))
    assert load_data(training_dir).sources['S1'].at['S1-5', 'business_name'] == 'NA'


def test_missing_target_is_not_silently_ignored(training_dir):
    data = load_data(training_dir)
    data.truth.loc['S1-1'] = 'S2-missing'
    with pytest.raises(ValueError, match='S2-missing is absent from S2'):
        evaluate(data, progress_every=0)


def test_duplicate_source_ids_rejected(training_dir):
    file = training_dir / 'train_source2.tsv'
    file.write_text(file.read_text().replace('S2-3', 'S2-1'))
    with pytest.raises(ValueError):
        load_data(training_dir)


def test_incomplete_truth_rejected(training_dir):
    file = training_dir / 'train_ground_truth.tsv'
    file.write_text(file.read_text().replace('S1-5', 'S1-other'))
    with pytest.raises(ValueError, match='exactly one row'):
        load_data(training_dir)


def test_cli_from_outside_repository(training_dir, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_m2.py'
    result = subprocess.run(
        [sys.executable, str(script), '--data-dir', str(training_dir),
         '--include-m1', '--progress-every', '0', '--missed-examples', '1'],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert 'TOTAL: 6' in result.stdout
    assert 'M1 union M2: recall=100.000000%; candidates=8' in result.stdout
    assert 'loading:' in result.stdout
    assert 'total (before report printing):' in result.stdout
    assert 'Example 2:' not in result.stdout


def test_empty_dataset_has_defined_report(training_dir):
    data = load_data(training_dir)
    data.sources = {source: frame.iloc[:0] for source, frame in data.sources.items()}
    data.truth = data.truth.iloc[:0]
    data.ground_truth_rows = 0
    report = evaluate(data, include_m1=True, progress_every=0)
    assert report['m2']['candidate_pairs'] == 0
    assert report['m2']['naive_pair_space'] == 0
    assert report['m2']['pair_recall'] is None
    assert report['m2']['candidate_reduction'] is None
    assert report['distributions']['Total per S1']['maximum'] == 0
    assert report['comparison']['M1 union M2']['pair_recall'] is None


@pytest.fixture
def multi_match_dir(tmp_path):
    """Hand-counted fixture: repeated targets across S1 rows are separate pairs.

    M2 candidate IDs by query (S2 | S3):
      A: 1,2,4 | 1,4    B: none | none
      C: 1,2,4 | 1,4    D: none | 5
    C has no labels but contributes five pairs. A -> S2-4 is a false candidate.
    M1: A -> S2-1, B -> S2-3/S3-3 (three pairs, all true).
    The only M1/M2 overlap is A -> S2-1, so union volume is 11 + 3 - 1.
    A has two S2 and three S3 true matches; asymmetric source totals detect swaps.
    """
    rows = {
        '1': [('S1-A', 'Alpha Ltd.'), ('S1-B', 'Private Limited'),
              ('S1-C', 'Alpha'), ('S1-D', 'Zeta')],
        '2': [('S2-1', 'Alpha Limited'), ('S2-2', 'Alpha Finance'),
              ('S2-3', 'Private Limited'), ('S2-4', 'Alpha Unrelated'),
              ('S2-5', 'Theta')],
        '3': [('S3-1', 'Alpha Trading'), ('S3-2', 'Delta'),
              ('S3-3', 'Private Limited'), ('S3-4', 'Alpha Beta'),
              ('S3-5', 'Zeta Corp')],
    }
    for source, records in rows.items():
        pd.DataFrame(records, columns=['entity_id', 'business_name']).to_csv(
            tmp_path / f'train_source{source}.tsv', sep='\t', index=False,
        )
    pd.DataFrame([
        ('S1-C', None), ('S1-D', 'S3-5,S2-5'),
        ('S1-A', 'S3-4,S2-2,S3-2,S2-1,S3-1,S2-1'),
        ('S1-B', 'S3-3,S2-3'),
    ], columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        tmp_path / 'train_ground_truth.tsv', sep='\t', index=False,
    )
    return tmp_path


def test_multi_match_exact_expected_metrics(multi_match_dir):
    report = evaluate(load_data(multi_match_dir), include_m1=True, progress_every=0)
    assert report['rows'] == {'S1': 4, 'S2': 5, 'S3': 5}
    assert report['rows_with_matches'] == 3
    assert report['rows_without_matches'] == 1
    assert report['m2'] == {
        'true_pairs': 9, 'captured_true_pairs': 5, 'missed_true_pairs': 4,
        'candidate_pairs': 11, 'pair_recall': 5 / 9,
        'naive_pair_space': 40, 'candidate_reduction': 1 - 11 / 40,
    }
    assert report['sources'] == {
        'S2': {'true_pairs': 4, 'captured_true_pairs': 2, 'missed_true_pairs': 2,
               'candidate_pairs': 6, 'pair_recall': 1 / 2},
        'S3': {'true_pairs': 5, 'captured_true_pairs': 3, 'missed_true_pairs': 2,
               'candidate_pairs': 5, 'pair_recall': 3 / 5},
    }
    assert report['comparison'] == {
        'M1': {'captured_true_pairs': 3, 'candidate_pairs': 3, 'pair_recall': 1 / 3},
        'M2': {'captured_true_pairs': 5, 'candidate_pairs': 11, 'pair_recall': 5 / 9},
        'M1 union M2': {'captured_true_pairs': 7, 'candidate_pairs': 13,
                        'pair_recall': 7 / 9},
    }
    for source, median, maximum in [('S1 -> S2', 1.5, 3), ('S1 -> S3', 1.5, 2),
                                    ('Total per S1', 3., 5)]:
        assert report['distributions'][source] == {
            '50%': median, '90%': maximum, '95%': maximum, '99%': maximum,
            '99.9%': maximum, 'maximum': maximum,
        }
    examples = report['missed_examples']
    assert [(e['S1 ID'], e['true target ID']) for e in examples] == [
        ('S1-A', 'S3-2'), ('S1-B', 'S2-3'), ('S1-B', 'S3-3'), ('S1-D', 'S2-5'),
    ]
    assert examples[0] == {
        'S1 ID': 'S1-A', 'S1 business name': 'Alpha Ltd.',
        'true target ID': 'S3-2', 'true target business name': 'Delta',
        'normalized S1 name': 'alpha limited', 'normalized target name': 'delta',
        'S1 informative tokens': ['alpha'], 'target informative tokens': ['delta'],
        'shared informative tokens': [],
    }
    assert all(set(example) == set(examples[0]) for example in examples)


@pytest.mark.parametrize('missing', [None, float('nan'), pd.NA, '', '  '])
def test_missing_truth_in_full_evaluation(multi_match_dir, missing):
    data = load_data(multi_match_dir)
    data.truth = data.truth.astype(object)
    data.truth.loc['S1-C'] = missing
    report = evaluate(data, include_m1=True, progress_every=0)
    assert report['m2']['true_pairs'] == 9
    assert report['m2']['captured_true_pairs'] == 5
    assert report['m2']['candidate_pairs'] == 11
    assert report['comparison']['M1 union M2']['candidate_pairs'] == 13


def test_raw_m2_configuration_candidate_lifetime_and_m1_reuse(multi_match_dir, monkeypatch):
    import weakref
    from unittest.mock import Mock

    import scripts.evaluate_m2 as evaluator

    data = load_data(multi_match_dir)
    original_generate = evaluator.generate_token_candidates
    build_spy = Mock(wraps=evaluator.build_token_index)
    normalize_spy = Mock(wraps=evaluator.normalize_name)
    previous = None
    calls = 0

    class CandidateList(list):
        """Weak-referenceable list to detect retention of previous query output."""

    def checked_generate(name, index, **kwargs):
        nonlocal previous, calls
        assert previous is None or previous() is None
        assert kwargs == {'min_token_length': 2, 'max_candidates': None}
        result = CandidateList(original_generate(name, index, **kwargs))
        previous = weakref.ref(result)
        calls += 1
        return result

    monkeypatch.setattr(evaluator, 'build_token_index', build_spy)
    monkeypatch.setattr(evaluator, 'generate_token_candidates', checked_generate)
    monkeypatch.setattr(evaluator, 'normalize_name', normalize_spy)
    report = evaluator.evaluate(data, include_m1=True, missed_examples=0, progress_every=0)
    assert calls == 8  # Both sources for all four queries, even the unlabeled one.
    assert previous() is None
    assert build_spy.call_count == 2
    for call in build_spy.call_args_list:
        assert call.kwargs == {'min_token_length': 2, 'max_token_frequency': None}
    # M1 normalizes each target once and each S1 once, shared across both sources.
    assert normalize_spy.call_count == 5 + 5 + 4
    assert report['comparison']['M1 union M2']['candidate_pairs'] == 13


def test_top_30_and_first_20_misses_are_deterministic(tmp_path):
    pd.DataFrame([('S1-1', 'absent')], columns=['entity_id', 'business_name']).to_csv(
        tmp_path / 'train_source1.tsv', sep='\t', index=False,
    )
    truth = []
    for source in ('2', '3'):
        records = [(f'S{source}-{i:02}', f'common token{i:02}') for i in range(35)]
        pd.DataFrame(records, columns=['entity_id', 'business_name']).to_csv(
            tmp_path / f'train_source{source}.tsv', sep='\t', index=False,
        )
        truth.extend(entity_id for entity_id, _ in records)
    pd.DataFrame([('S1-1', ','.join(reversed(truth)))],
                 columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        tmp_path / 'train_ground_truth.tsv', sep='\t', index=False,
    )
    data = load_data(tmp_path)
    first = evaluate(data, progress_every=0)
    second = evaluate(data, progress_every=0)
    assert first['m2']['true_pairs'] == 70
    assert first['m2']['captured_true_pairs'] == 0
    assert first['m2']['candidate_pairs'] == 0
    assert first['missed_examples'] == second['missed_examples']
    assert [e['true target ID'] for e in first['missed_examples']] == [
        f'S2-{i:02}' for i in range(20)
    ]
    for source in ('S2', 'S3'):
        assert first['tokens'][source]['indexed_tokens'] == 36
        assert first['tokens'][source]['top_tokens'] == [
            ('common', 35), *[(f'token{i:02}', 1) for i in range(29)],
        ]
