"""动态语义质量门槛测试。"""


def apply_threshold(scores, absolute_min=0.5, relative_ratio=0.7):
    best = max(scores)
    threshold = max(absolute_min, best * relative_ratio)
    return threshold, [score for score in scores if score >= threshold]


def test_absolute_threshold_removes_low_scores():
    threshold, valid = apply_threshold([0.76, 0.53, 0.45])
    assert abs(threshold - 0.532) < 1e-9
    assert valid == [0.76]


def test_relative_threshold_changes_with_best_score():
    threshold, valid = apply_threshold([0.94, 0.70, 0.65, 0.50])
    assert abs(threshold - 0.658) < 1e-9
    assert valid == [0.94, 0.70]
