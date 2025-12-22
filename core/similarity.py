"""
相似度计算模块
处理频道相似度评分、标签匹配等
"""
import math
from typing import Any, Dict

import numpy as np

from infrastructure.config import Config


def scale_score(base_subs: int, candidate_subs: int) -> float:
    """
    量级相近度：订阅数数量级越接近，得分越高（0~1）。
    使用 log10 以避免大号把差距拉得太夸张。
    
    注意：根据 SimilarTube 的实际排序，订阅数不是主要排序因素，
    即使很小的频道（如 762 订阅）只要 Topics/Audience 匹配度高也能排在前面。
    所以这里只做轻微的"量级相近度"加分，不作为主要排序依据。
    
    Args:
        base_subs: 基频道订阅数
        candidate_subs: 候选频道订阅数
    
    Returns:
        量级相近度得分（0-1）
    """
    if base_subs <= 0 or candidate_subs <= 0:
        return 0.5  # 没有数据时给一个中性分

    diff = abs(math.log10(candidate_subs + 1) - math.log10(base_subs + 1))
    # diff=0 -> 1 分；diff>=max_diff（相差 1000 倍）-> 0 分，中间线性下降
    max_diff = Config.SCALE_SCORE["max_diff"]
    return max(0.0, 1.0 - diff / max_diff)


def calculate_tag_overlap(base_tags: set[str], candidate_tags: set[str]) -> float:
    """
    计算标签重合度（Jaccard 相似度）。
    
    Args:
        base_tags: 基频道标签集合
        candidate_tags: 候选频道标签集合
    
    Returns:
        标签重合度（0-1）
    """
    if not base_tags or not candidate_tags:
        return 0.0
    intersection = len(base_tags & candidate_tags)
    union = len(base_tags | candidate_tags)
    if union == 0:
        return 0.0
    return intersection / union


def calculate_total_score(
    semantic_sim: float,
    tag_score: float,
    scale_score_val: float,
) -> float:
    """
    计算综合得分。
    
    根据 SimilarTube 的实际排序，Topics/Audience 标签匹配度是最重要的因素。
    权重分配：标签相似度 45% + 语义相似度 40% + 订阅量级 15%
    
    Args:
        semantic_sim: 语义相似度
        tag_score: 标签相似度
        scale_score_val: 订阅量级得分
    
    Returns:
        综合得分
    """
    weights = Config.SIMILARITY_WEIGHTS
    return (
        weights["tag_score"] * tag_score
        + weights["semantic_sim"] * semantic_sim
        + weights["scale_score"] * scale_score_val
    )

