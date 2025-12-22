"""
数据验证模块
统一管理数据验证逻辑，提供通用的验证函数
"""
import re
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger()


def validate_channel_id(channel_id: str) -> bool:
    """
    验证频道ID格式
    
    Args:
        channel_id: 频道ID字符串
        
    Returns:
        如果格式有效返回True，否则返回False
    """
    if not channel_id or not isinstance(channel_id, str):
        return False
    
    # YouTube频道ID格式：UC开头的24个字符，或自定义频道名
    channel_id = channel_id.strip()
    if len(channel_id) < 1 or len(channel_id) > 100:
        return False
    
    # 基本格式检查：允许字母、数字、连字符、下划线
    if not re.match(r'^[a-zA-Z0-9_-]+$', channel_id):
        return False
    
    return True


def validate_channel_url(url: str) -> bool:
    """
    验证YouTube频道URL格式
    
    Args:
        url: 频道URL字符串
        
    Returns:
        如果格式有效返回True，否则返回False
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # 支持的URL格式
    patterns = [
        r'^https?://(www\.)?youtube\.com/channel/[a-zA-Z0-9_-]+',
        r'^https?://(www\.)?youtube\.com/c/[a-zA-Z0-9_-]+',
        r'^https?://(www\.)?youtube\.com/user/[a-zA-Z0-9_-]+',
        r'^https?://(www\.)?youtube\.com/@[a-zA-Z0-9_-]+',
        r'^https?://youtu\.be/[a-zA-Z0-9_-]+',
    ]
    
    for pattern in patterns:
        if re.match(pattern, url):
            return True
    
    return False


def validate_subscriber_count(count: Any) -> bool:
    """
    验证订阅数是否有效
    
    Args:
        count: 订阅数（可以是int、str等）
        
    Returns:
        如果有效返回True，否则返回False
    """
    if count is None:
        return True  # None表示不限制
    
    try:
        count_int = int(count)
        return count_int >= 0
    except (ValueError, TypeError):
        return False


def validate_similarity_score(score: Any) -> bool:
    """
    验证相似度分数是否有效
    
    Args:
        score: 相似度分数（0-1之间）
        
    Returns:
        如果有效返回True，否则返回False
    """
    if score is None:
        return True  # None表示不限制
    
    try:
        score_float = float(score)
        return 0.0 <= score_float <= 1.0
    except (ValueError, TypeError):
        return False


def validate_max_results(max_results: Any) -> bool:
    """
    验证最大结果数是否有效
    
    Args:
        max_results: 最大结果数
        
    Returns:
        如果有效返回True，否则返回False
    """
    try:
        max_int = int(max_results)
        return 1 <= max_int <= 200
    except (ValueError, TypeError):
        return False


def validate_language_code(language: Optional[str]) -> bool:
    """
    验证语言代码格式
    
    Args:
        language: 语言代码（如 "en", "zh-Hans"）
        
    Returns:
        如果格式有效返回True，否则返回False
    """
    if language is None:
        return True  # None表示不限制
    
    if not isinstance(language, str):
        return False
    
    language = language.strip()
    if not language:
        return True  # 空字符串视为不限制
    
    # 基本格式检查：允许字母、连字符
    if not re.match(r'^[a-zA-Z-]+$', language):
        return False
    
    return True


def validate_region_code(region: Optional[str]) -> bool:
    """
    验证地区代码格式
    
    Args:
        region: 地区代码（如 "US", "CN"）
        
    Returns:
        如果格式有效返回True，否则返回False
    """
    if region is None:
        return True  # None表示不限制
    
    if not isinstance(region, str):
        return False
    
    region = region.strip().upper()
    if not region:
        return True  # 空字符串视为不限制
    
    # ISO 3166-1 alpha-2 格式：2个大写字母
    if not re.match(r'^[A-Z]{2}$', region):
        return False
    
    return True


def validate_channel_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    验证频道数据字典的完整性和有效性
    
    Args:
        data: 频道数据字典
        
    Returns:
        如果验证通过，返回清理后的数据字典；否则返回None
    """
    if not isinstance(data, dict):
        return None
    
    # 验证必需的字段
    channel_id = data.get("channelId") or data.get("channel_id")
    if not channel_id or not validate_channel_id(str(channel_id)):
        logger.debug(f"频道数据验证失败：无效的频道ID: {channel_id}")
        return None
    
    # 清理和验证数值字段
    validated_data = {
        "channelId": str(channel_id).strip(),
        "title": str(data.get("title", "")).strip()[:500],  # 限制长度
        "description": str(data.get("description", "")).strip()[:10000],  # 限制长度
    }
    
    # 验证订阅数
    try:
        sub_count = int(data.get("subscriberCount") or data.get("subscriber_count") or 0)
        validated_data["subscriberCount"] = max(0, sub_count)
    except (ValueError, TypeError):
        validated_data["subscriberCount"] = 0
    
    # 验证观看数
    try:
        view_count = int(data.get("viewCount") or data.get("view_count") or 0)
        validated_data["viewCount"] = max(0, view_count)
    except (ValueError, TypeError):
        validated_data["viewCount"] = 0
    
    # 验证列表字段
    for field in ["emails", "topics", "audience"]:
        value = data.get(field, [])
        if isinstance(value, list):
            validated_data[field] = value
        else:
            validated_data[field] = []
    
    # 验证可选字段
    for field in ["country", "defaultLanguage", "language"]:
        value = data.get(field)
        if value is not None:
            validated_data[field] = str(value).strip()[:100]
        else:
            validated_data[field] = None
    
    return validated_data


def validate_search_request(
    channel_url: str,
    max_results: int = 20,
    min_subscribers: Optional[int] = None,
    max_subscribers: Optional[int] = None,
    preferred_language: Optional[str] = None,
    preferred_region: Optional[str] = None,
    min_similarity: Optional[float] = None,
) -> tuple[bool, Optional[str]]:
    """
    验证搜索请求的所有参数
    
    Args:
        channel_url: 频道URL
        max_results: 最大结果数
        min_subscribers: 最小订阅数
        max_subscribers: 最大订阅数
        preferred_language: 偏好语言
        preferred_region: 偏好地区
        min_similarity: 最小相似度
        
    Returns:
        (is_valid, error_message) 元组
    """
    # 验证频道URL
    if not validate_channel_url(channel_url):
        return False, "无效的频道URL格式"
    
    # 验证最大结果数
    if not validate_max_results(max_results):
        return False, "max_results 必须在 1-200 之间"
    
    # 验证订阅数范围
    if min_subscribers is not None and not validate_subscriber_count(min_subscribers):
        return False, "min_subscribers 不能为负数"
    
    if max_subscribers is not None and not validate_subscriber_count(max_subscribers):
        return False, "max_subscribers 不能为负数"
    
    if min_subscribers is not None and max_subscribers is not None:
        if min_subscribers > max_subscribers:
            return False, "min_subscribers 不能大于 max_subscribers"
    
    # 验证语言代码
    if preferred_language is not None and not validate_language_code(preferred_language):
        return False, "无效的语言代码格式"
    
    # 验证地区代码
    if preferred_region is not None and not validate_region_code(preferred_region):
        return False, "无效的地区代码格式（应为ISO 3166-1 alpha-2格式，如US、CN）"
    
    # 验证相似度
    if min_similarity is not None and not validate_similarity_score(min_similarity):
        return False, "min_similarity 必须在 0-1 之间"
    
    return True, None

