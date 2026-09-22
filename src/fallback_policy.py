"""三级检索降级策略。"""


def needs_fallback(candidate_count: int, required_count: int = 3) -> bool:
    """候选少于目标数量时才允许调用下一层网络引擎。"""
    if required_count < 1:
        raise ValueError("required_count必须大于0")
    return candidate_count < required_count
