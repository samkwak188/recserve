import numpy as np
from scripts.quality_production import boundaries, examples, confidence


def test_global_cutoffs_do_not_split_timestamp_ties():
    raw = np.array([[1, i, 4, i // 2] for i in range(100)])
    train, validation = boundaries(raw)
    training = raw[raw[:, 3] <= train]
    valid = raw[(raw[:, 3] > train) & (raw[:, 3] <= validation)]
    test = raw[raw[:, 3] > validation]
    assert training[:, 3].max() < valid[:, 3].min()
    assert valid[:, 3].max() < test[:, 3].min()


def test_onboarding_never_contains_future_targets_and_reports_cold_items():
    raw = np.array([[5, i, 5, i] for i in range(1, 12)])
    rows, omitted = examples(raw, 4, 11, set(range(1, 11)), set(), 20)
    user, cohort, prefs, gold = rows[0]
    assert cohort == 'held_out_user'
    assert set(prefs) == set(range(5, 10)) and gold == {10}
    assert set(prefs).isdisjoint(gold)
    assert omitted['unsupported_target_items'] == 1


def test_bootstrap_is_paired_and_repeatable():
    assert confidence([.2, .3, .4], [.1, .2, .3])[0] > 0
    assert confidence([.2, .3], [.2, .3]) == [0., 0.]
