from threshold import choose_threshold, entity_decision, keep_match


def test_choose_threshold_uses_probabilities_with_labels():
    labels = [0, 0, 1, 1]
    probs = [0.1, 0.45, 0.55, 0.9]

    threshold = choose_threshold(labels, probs)

    assert 0.45 <= threshold <= 0.9


def test_keep_match_uses_threshold_on_probability():
    assert keep_match(0.51, 0.5) is True
    assert keep_match(0.49, 0.5) is False


def test_entity_decision_keeps_singletons_and_uses_max_pair_score():
    assert entity_decision([], singleton=True) is True
    assert entity_decision([0.2, 0.3], threshold=0.5) is False
    assert entity_decision([0.2, 0.8], threshold=0.5) is True
