"""
BD评分模块（交易所BD寻找KOL专用）

提供针对交易所BD寻找合约返佣KOL的专业评分功能：
- 合约内容聚焦度评分
- 受众质量评分
- 商业化潜力评分
- 竞品合作检测
- BD综合评分
"""
import re
from typing import Any, Dict, List, Tuple

from config import Config
from logger import get_logger

logger = get_logger()


def detect_competitor_collaborations(
    description: str,
    recent_videos: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    检测频道是否与竞品交易所有返佣合作。
    
    通过分析频道描述和最近视频描述中的返佣链接来判断。
    
    Args:
        description: 频道描述
        recent_videos: 最近视频列表
    
    Returns:
        {
            "has_competitor_collab": bool,
            "competitors": List[str],  # 检测到的竞品列表
            "competitor_details": Dict[str, List[str]],  # 每个竞品匹配到的模式
        }
    """
    if recent_videos is None:
        recent_videos = []
    
    # 合并所有文本
    all_text = description.lower()
    for video in recent_videos:
        if video and isinstance(video, dict):
            video_desc = video.get("description", "")
            if video_desc:
                all_text += " " + video_desc.lower()
    
    found_competitors: Dict[str, List[str]] = {}
    
    for exchange, patterns in Config.COMPETITOR_EXCHANGES.items():
        matched_patterns = []
        for pattern in patterns:
            if pattern.lower() in all_text:
                matched_patterns.append(pattern)
        
        if matched_patterns:
            found_competitors[exchange] = matched_patterns
    
    return {
        "has_competitor_collab": len(found_competitors) > 0,
        "competitors": list(found_competitors.keys()),
        "competitor_details": found_competitors,
    }


def calculate_contract_focus_score(
    topics: List[str],
    description: str = "",
    recent_videos: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    计算频道对合约交易内容的聚焦程度。
    
    基于：
    1. Topics标签中合约相关标签的占比和权重
    2. 频道描述中合约相关关键词的出现
    3. 最近视频标题中的合约相关关键词
    
    Args:
        topics: 频道的Topics标签列表
        description: 频道描述
        recent_videos: 最近视频列表
    
    Returns:
        {
            "score": float (0-1),
            "high_value_topics": List[str],  # 命中的高价值标签
            "negative_topics": List[str],    # 命中的负面标签
            "keyword_matches": List[str],    # 描述/标题中的关键词匹配
        }
    """
    if recent_videos is None:
        recent_videos = []
    
    # 1. 基于Topics标签计算
    weighted_score = 0.0
    topic_count = 0
    high_value_topics = []
    negative_topics = []
    
    for topic in topics:
        topic_lower = topic.lower()
        weight = Config.BD_TOPIC_WEIGHTS.get(topic_lower, 0.0)
        
        if weight > 0.5:
            high_value_topics.append(topic)
        elif weight < 0:
            negative_topics.append(topic)
        
        weighted_score += weight
        topic_count += 1
    
    # 归一化标签得分
    if topic_count > 0:
        topic_score = max(0.0, min(1.0, (weighted_score / topic_count + 0.3) / 1.3))
    else:
        topic_score = 0.0
    
    # 2. 基于关键词的额外得分
    contract_keywords = [
        "futures", "leverage", "perpetual", "margin", "合约", "杠杆",
        "做多", "做空", "long", "short", "liquidation", "爆仓",
        "funding rate", "资金费率", "开仓", "平仓", "止损", "止盈",
        "scalping", "scalp", "倍", "x leverage", "x杠杆",
    ]
    
    all_text = description.lower()
    for video in recent_videos:
        if video and isinstance(video, dict):
            video_title = video.get("title", "")
            video_desc = video.get("description", "")
            all_text += f" {video_title} {video_desc}".lower()
    
    keyword_matches = []
    for keyword in contract_keywords:
        if keyword.lower() in all_text:
            keyword_matches.append(keyword)
    
    # 关键词得分（最多+0.3）
    keyword_score = min(0.3, len(keyword_matches) * 0.05)
    
    # 综合得分
    final_score = min(1.0, topic_score * 0.7 + keyword_score + 0.1)  # 0.1 基础分
    
    return {
        "score": round(final_score, 3),
        "topic_score": round(topic_score, 3),
        "keyword_score": round(keyword_score, 3),
        "high_value_topics": high_value_topics,
        "negative_topics": negative_topics,
        "keyword_matches": keyword_matches[:10],  # 最多返回10个
    }


def calculate_audience_quality_score(
    audience: List[str],
    subscriber_count: int = 0,
    engagement_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    计算受众质量评分（是否为真实交易者）。
    
    Args:
        audience: 受众标签列表
        subscriber_count: 订阅数
        engagement_rate: 互动率
    
    Returns:
        {
            "score": float (0-1),
            "high_value_audience": List[str],
            "low_value_audience": List[str],
            "engagement_quality": str,  # excellent/good/average/poor
        }
    """
    # 1. 基于受众标签计算
    weighted_score = 0.0
    audience_count = 0
    high_value_audience = []
    low_value_audience = []
    
    for aud in audience:
        aud_lower = aud.lower()
        weight = Config.BD_AUDIENCE_WEIGHTS.get(aud_lower, 0.3)
        
        if weight >= 0.8:
            high_value_audience.append(aud)
        elif weight <= 0.2:
            low_value_audience.append(aud)
        
        weighted_score += weight
        audience_count += 1
    
    if audience_count > 0:
        audience_score = weighted_score / audience_count
    else:
        audience_score = 0.3  # 默认中等
    
    # 2. 互动率质量评估
    engagement_quality = "poor"
    engagement_bonus = 0.0
    
    if engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["excellent"]:
        engagement_quality = "excellent"
        engagement_bonus = 0.2
    elif engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["good"]:
        engagement_quality = "good"
        engagement_bonus = 0.15
    elif engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["average"]:
        engagement_quality = "average"
        engagement_bonus = 0.05
    else:
        engagement_quality = "poor"
        engagement_bonus = -0.1
    
    # 综合得分
    final_score = max(0.0, min(1.0, audience_score * 0.7 + engagement_bonus + 0.15))
    
    return {
        "score": round(final_score, 3),
        "audience_score": round(audience_score, 3),
        "high_value_audience": high_value_audience,
        "low_value_audience": low_value_audience,
        "engagement_quality": engagement_quality,
        "engagement_bonus": round(engagement_bonus, 3),
    }


def calculate_commercialization_score(
    emails: List[str],
    description: str,
    recent_videos: List[Dict[str, Any]] | None = None,
    competitor_result: Dict[str, Any] | None = None,
    engagement_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    计算商业化潜力评分。
    
    评估KOL的商业化意愿和合作可能性。
    
    Args:
        emails: 联系邮箱列表
        description: 频道描述
        recent_videos: 最近视频列表
        competitor_result: 竞品检测结果（可选，避免重复检测）
        engagement_rate: 互动率
    
    Returns:
        {
            "score": float (0-1),
            "has_email": bool,
            "has_collab_experience": bool,
            "has_collab_keywords": bool,
            "signals": List[str],  # 商业化信号
            "concerns": List[str],  # 潜在问题
        }
    """
    if recent_videos is None:
        recent_videos = []
    
    score = 0.0
    signals = []
    concerns = []
    
    # 1. 是否有联系邮箱（重要，+0.25）
    has_email = len(emails) > 0
    if has_email:
        score += 0.25
        signals.append(f"有联系邮箱 ({len(emails)}个)")
    else:
        concerns.append("无公开联系邮箱")
    
    # 2. 是否有竞品合作经验（说明有合作意愿，+0.2）
    if competitor_result is None:
        competitor_result = detect_competitor_collaborations(description, recent_videos)
    
    has_collab_experience = competitor_result.get("has_competitor_collab", False)
    if has_collab_experience:
        competitors = competitor_result.get("competitors", [])
        score += 0.2
        signals.append(f"已与 {', '.join(competitors[:3])} 合作")
        if len(competitors) >= 3:
            concerns.append(f"已与{len(competitors)}家交易所合作，可能有排他协议")
    else:
        signals.append("无检测到现有交易所合作")
    
    # 3. 描述中是否提及合作/赞助（+0.15）
    all_text = description.lower()
    for video in recent_videos:
        if video and isinstance(video, dict):
            video_desc = video.get("description", "")
            all_text += " " + video_desc.lower()
    
    has_collab_keywords = False
    matched_keywords = []
    for keyword in Config.BD_COLLAB_KEYWORDS:
        if keyword.lower() in all_text:
            has_collab_keywords = True
            matched_keywords.append(keyword)
    
    if has_collab_keywords:
        score += 0.15
        signals.append(f"描述中提及: {', '.join(matched_keywords[:3])}")
    
    # 4. 互动率评估（+0.2）
    if engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["excellent"]:
        score += 0.2
        signals.append(f"高互动率 {engagement_rate:.1f}%")
    elif engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["good"]:
        score += 0.15
        signals.append(f"良好互动率 {engagement_rate:.1f}%")
    elif engagement_rate >= Config.BD_ENGAGEMENT_THRESHOLDS["average"]:
        score += 0.1
        signals.append(f"中等互动率 {engagement_rate:.1f}%")
    else:
        concerns.append(f"低互动率 {engagement_rate:.1f}%")
    
    # 5. 是否有专业社交媒体链接（+0.1）
    social_patterns = [
        r"twitter\.com|x\.com",
        r"t\.me|telegram",
        r"discord\.gg|discord\.com",
    ]
    social_found = []
    for pattern in social_patterns:
        if re.search(pattern, all_text, re.IGNORECASE):
            platform = pattern.split(r"|")[0].replace(r"\.", ".")
            social_found.append(platform)
    
    if social_found:
        score += 0.1
        signals.append(f"有社交媒体链接: {', '.join(social_found[:3])}")
    
    return {
        "score": round(min(1.0, score), 3),
        "has_email": has_email,
        "emails": emails[:5],  # 最多返回5个
        "has_collab_experience": has_collab_experience,
        "competitors": competitor_result.get("competitors", []),
        "has_collab_keywords": has_collab_keywords,
        "signals": signals,
        "concerns": concerns,
    }


def calculate_bd_total_score(
    contract_focus: Dict[str, Any],
    audience_quality: Dict[str, Any],
    commercialization: Dict[str, Any],
    semantic_sim: float = 0.0,
    scale_score: float = 0.0,
) -> Dict[str, Any]:
    """
    计算BD综合评分。
    
    基于各维度评分和权重计算综合得分，并给出优先级建议。
    
    Args:
        contract_focus: 合约聚焦度评分结果
        audience_quality: 受众质量评分结果
        commercialization: 商业化潜力评分结果
        semantic_sim: 语义相似度
        scale_score: 规模得分
    
    Returns:
        {
            "total_score": float (0-1),
            "priority": str,  # high/medium/low
            "breakdown": Dict,  # 各维度得分明细
            "recommendation": Dict,  # 合作建议
        }
    """
    weights = Config.BD_SIMILARITY_WEIGHTS
    
    # 获取各维度分数
    contract_score = contract_focus.get("score", 0.0)
    audience_score = audience_quality.get("score", 0.0)
    commercial_score = commercialization.get("score", 0.0)
    engagement_quality = audience_quality.get("engagement_quality", "average")
    
    # 互动率分数映射
    engagement_score_map = {
        "excellent": 1.0,
        "good": 0.75,
        "average": 0.5,
        "poor": 0.2,
    }
    engagement_score = engagement_score_map.get(engagement_quality, 0.5)
    
    # 加权计算总分
    total_score = (
        weights["contract_focus_score"] * contract_score +
        weights["audience_quality_score"] * audience_score +
        weights["commercialization_score"] * commercial_score +
        weights["engagement_rate_score"] * engagement_score +
        weights["semantic_sim"] * semantic_sim +
        weights["scale_score"] * scale_score
    )
    
    # 确定优先级
    if total_score >= Config.BD_PRIORITY_THRESHOLDS["high"]:
        priority = "high"
    elif total_score >= Config.BD_PRIORITY_THRESHOLDS["medium"]:
        priority = "medium"
    elif total_score >= Config.BD_PRIORITY_THRESHOLDS["low"]:
        priority = "low"
    else:
        priority = "skip"  # 不建议联系
    
    # 生成合作建议
    reasons = []
    concerns = []
    
    # 收集正面理由
    if contract_score >= 0.6:
        reasons.append("专注合约交易内容")
    if len(contract_focus.get("high_value_topics", [])) >= 2:
        topics = ", ".join(contract_focus.get("high_value_topics", [])[:3])
        reasons.append(f"核心标签: {topics}")
    
    if audience_score >= 0.6:
        reasons.append("受众为真实交易者")
    if len(audience_quality.get("high_value_audience", [])) >= 1:
        aud = ", ".join(audience_quality.get("high_value_audience", [])[:2])
        reasons.append(f"目标受众: {aud}")
    
    if commercial_score >= 0.5:
        reasons.append("商业化潜力较高")
    if commercialization.get("has_email"):
        reasons.append("有公开联系方式")
    if commercialization.get("has_collab_experience"):
        reasons.append("有返佣合作经验")
    
    if engagement_quality in ["excellent", "good"]:
        reasons.append(f"互动率{engagement_quality}")
    
    # 收集潜在问题
    concerns.extend(commercialization.get("concerns", []))
    
    if len(contract_focus.get("negative_topics", [])) >= 2:
        concerns.append("频道内容与合约交易关联度较低")
    
    if len(audience_quality.get("low_value_audience", [])) >= 2:
        concerns.append("受众可能不是活跃交易者")
    
    return {
        "total_score": round(total_score, 3),
        "priority": priority,
        "breakdown": {
            "contract_focus_score": round(contract_score, 3),
            "audience_quality_score": round(audience_score, 3),
            "commercialization_score": round(commercial_score, 3),
            "engagement_rate_score": round(engagement_score, 3),
            "semantic_sim": round(semantic_sim, 3),
            "scale_score": round(scale_score, 3),
        },
        "recommendation": {
            "priority": priority,
            "reasons": reasons[:5],  # 最多5条
            "concerns": concerns[:3],  # 最多3条
            "action": _get_action_suggestion(priority, commercialization),
        },
    }


def _get_action_suggestion(
    priority: str,
    commercialization: Dict[str, Any],
) -> str:
    """
    根据优先级和商业化信息生成行动建议。
    """
    has_email = commercialization.get("has_email", False)
    has_collab = commercialization.get("has_collab_experience", False)
    competitors = commercialization.get("competitors", [])
    
    if priority == "high":
        if has_email:
            return "建议立即发送合作邮件，提供有竞争力的返佣比例"
        else:
            return "高价值目标，建议通过社交媒体私信联系"
    elif priority == "medium":
        if has_collab and len(competitors) >= 2:
            return "已有多家合作，建议提供差异化条件"
        elif has_email:
            return "可加入联系清单，定期跟进"
        else:
            return "可关注，待有联系方式后跟进"
    elif priority == "low":
        return "优先级较低，可放入后备清单"
    else:
        return "不建议联系，内容与合约交易关联度低"


def calculate_full_bd_metrics(
    channel_info: Dict[str, Any],
    semantic_sim: float = 0.0,
    scale_score_val: float = 0.0,
) -> Dict[str, Any]:
    """
    计算完整的BD评分指标（便捷封装函数）。
    
    Args:
        channel_info: 频道信息字典，应包含：
            - topics: List[str]
            - audience: List[str]
            - emails: List[str]
            - description: str
            - recent_videos: List[Dict]
            - engagement_rate: float
            - subscriberCount: int
        semantic_sim: 语义相似度
        scale_score_val: 规模得分
    
    Returns:
        完整的BD评分结果
    """
    # 提取频道信息
    topics = channel_info.get("topics", [])
    audience = channel_info.get("audience", [])
    emails = channel_info.get("emails", [])
    description = channel_info.get("description", "")
    recent_videos = channel_info.get("recent_videos", [])
    engagement_rate = channel_info.get("engagement_rate", 0.0)
    subscriber_count = channel_info.get("subscriberCount", 0)
    
    # 1. 竞品检测
    competitor_result = detect_competitor_collaborations(description, recent_videos)
    
    # 2. 合约聚焦度
    contract_focus = calculate_contract_focus_score(topics, description, recent_videos)
    
    # 3. 受众质量
    audience_quality = calculate_audience_quality_score(
        audience, subscriber_count, engagement_rate
    )
    
    # 4. 商业化潜力
    commercialization = calculate_commercialization_score(
        emails, description, recent_videos, competitor_result, engagement_rate
    )
    
    # 5. BD综合评分
    bd_total = calculate_bd_total_score(
        contract_focus, audience_quality, commercialization,
        semantic_sim, scale_score_val
    )
    
    return {
        "bd_total_score": bd_total["total_score"],
        "bd_priority": bd_total["priority"],
        "bd_metrics": {
            "contract_focus": contract_focus,
            "audience_quality": audience_quality,
            "commercialization": commercialization,
            "competitor_detection": competitor_result,
        },
        "bd_breakdown": bd_total["breakdown"],
        "bd_recommendation": bd_total["recommendation"],
    }

