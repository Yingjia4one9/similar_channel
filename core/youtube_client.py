"""
YouTube 相似频道查找主模块
提供高层 API，整合各个功能模块
"""
import concurrent.futures
import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from infrastructure.cache import invalidate_all_channel_caches
from core.candidate_collection import (
    collect_candidate_channels_from_related_videos,
    search_candidate_channels_by_title,
)
from core.channel_info import (
    batch_get_channels_info,
    get_channel_basic_info,
    get_recent_video_ids,
    get_recent_video_snippets_for_channel,
    get_recent_videos_stats,
)
from core.channel_parser import extract_channel_id_from_url
from infrastructure.config import Config
from infrastructure.database import (
    get_candidates_from_local_index,
    get_channel_basic_info_for_filtering,
    get_channel_info_from_local_db,
    get_embeddings_from_local_db,
    get_single_channel_info_from_local_db,
)
from core.embedding import (
    cosine_similarity,
    ensure_label_embeddings,
    get_embed_model,
    infer_topics_and_audience,
)
from core.similarity import (
    calculate_tag_overlap,
    calculate_total_score,
    scale_score,
)
from infrastructure.logger import get_logger
from infrastructure.utils import build_text_for_channel, extract_emails_from_text
from core.youtube_api import YouTubeQuotaExceededError
from infrastructure.quota_tracker import record_fallback_usage
from core.bd_scoring import calculate_full_bd_metrics

# 导入数据库保存函数（延迟导入，避免循环依赖）
try:
    from build_channel_index import _batch_upsert_channels
    DB_SAVE_AVAILABLE = True
except ImportError:
    DB_SAVE_AVAILABLE = False
    logger.warning("无法导入数据库保存函数，搜索过程中不会自动保存频道到数据库")

logger = get_logger()

# 导出异常类，保持向后兼容
__all__ = ["get_similar_channels_by_url", "YouTubeQuotaExceededError"]


# ==================== 重构后的辅助函数（CP-y6-03：函数职责单一） ====================

def _validate_search_params(
    channel_url: str,
    max_results: int,
    min_subscribers: int | None,
    max_subscribers: int | None,
    min_similarity: float | None,
) -> str:
    """
    验证搜索参数
    
    Args:
        channel_url: 频道链接
        max_results: 最大结果数
        min_subscribers: 最小订阅数
        max_subscribers: 最大订阅数
        min_similarity: 最低相似度
    
    Returns:
        清理后的频道链接
    
    Raises:
        ValueError: 如果参数无效
    """
    if not channel_url or not isinstance(channel_url, str) or not channel_url.strip():
        raise ValueError("频道链接不能为空")
    
    channel_url = channel_url.strip()
    
    if max_results < 1 or max_results > 200:
        raise ValueError("max_results 必须在 1-200 之间")
    
    if min_subscribers is not None and min_subscribers < 0:
        raise ValueError("min_subscribers 不能为负数")
    
    if max_subscribers is not None and max_subscribers < 0:
        raise ValueError("max_subscribers 不能为负数")
    
    if min_subscribers is not None and max_subscribers is not None:
        if min_subscribers > max_subscribers:
            raise ValueError("min_subscribers 不能大于 max_subscribers")
    
    if min_similarity is not None and (min_similarity < 0 or min_similarity > 1):
        raise ValueError("min_similarity 必须在 0-1 之间")
    
    return channel_url


def _get_base_channel_info(
    channel_id: str,
    report_progress: Callable[[float, str], None],
) -> Dict[str, Any]:
    """
    获取基频道信息（带降级策略）
    
    Args:
        channel_id: 频道ID
        report_progress: 进度报告函数
    
    Returns:
        基频道信息字典
    
    Raises:
        ValueError: 如果无法获取频道信息
    """
    report_progress(10, "正在获取频道基础信息...")
    try:
        base_info = get_channel_basic_info(channel_id, use_for="search")
    except YouTubeQuotaExceededError:
        logger.warning(f"API配额用尽，尝试从本地数据库读取频道信息: {channel_id}")
        local_base_info = get_single_channel_info_from_local_db(channel_id)
        if local_base_info:
            logger.info(f"成功从本地数据库获取频道信息: {channel_id}")
            base_info = local_base_info
            record_fallback_usage(
                fallback_type="local_db_channel_info",
                endpoint="channels",
                context=f"channel_id={channel_id}",
                success=True,
            )
        else:
            record_fallback_usage(
                fallback_type="local_db_channel_info",
                endpoint="channels",
                context=f"channel_id={channel_id}",
                success=False,
            )
            raise ValueError(
                "API配额已用尽，且本地数据库中没有该频道的信息。\n"
                "解决方案：\n"
                "1. 请等待配额重置（通常在每天UTC 00:00重置）\n"
                "2. 使用 build_channel_index.py 预先建立频道索引\n"
                "3. 或申请更高的API配额"
            )
    
    return base_info


def _enrich_base_channel_info(
    base_info: Dict[str, Any],
    channel_id: str,
    report_progress: Callable[[float, str], None],
) -> None:
    """
    丰富基频道信息（获取最近视频、提取邮箱等）
    
    Args:
        base_info: 基频道信息字典（会被修改）
        channel_id: 频道ID
        report_progress: 进度报告函数
    """
    report_progress(15, "正在获取最近视频信息...")
    base_recent_videos: List[Dict[str, Any]] = []
    try:
        base_recent_videos = get_recent_video_snippets_for_channel(
            channel_id, max_results=Config.CHANNEL_INFO["recent_videos_count"], use_for="search"
        )
        base_info["recent_videos"] = base_recent_videos
    except YouTubeQuotaExceededError:
        logger.warning(f"获取基频道视频信息时配额用尽，使用已有信息继续")
        base_recent_videos = base_info.get("recent_videos", [])
        base_info["recent_videos"] = base_recent_videos
        record_fallback_usage(
            fallback_type="skip_operation",
            endpoint="playlistItems",
            context=f"base_channel_videos channel_id={channel_id}",
            success=True,
        )
    
    # 提取邮箱
    base_emails: List[str] = []
    base_description = base_info.get("description")
    if base_description:
        base_emails.extend(extract_emails_from_text(base_description))
    if base_recent_videos:
        for v in base_recent_videos:
            if v and isinstance(v, dict):
                video_desc = v.get("description")
                if video_desc:
                    base_emails.extend(extract_emails_from_text(video_desc))
    base_info["emails"] = list(dict.fromkeys(base_emails)) if base_emails else []


def _compute_base_channel_embedding(
    base_info: Dict[str, Any],
    report_progress: Callable[[float, str], None],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    计算基频道的向量和标签
    
    Args:
        base_info: 基频道信息字典（会被修改，添加topics和audience）
        report_progress: 进度报告函数
    
    Returns:
        (base_vec, base_tags) 元组
    
    Raises:
        ValueError: 如果无法计算向量
    """
    report_progress(20, "正在计算频道向量和标签...")
    model = get_embed_model()
    ensure_label_embeddings(model)
    
    base_text = build_text_for_channel(base_info)
    if not base_text:
        raise ValueError("无法构建频道文本特征，频道信息可能不完整")
    
    base_vec = model.encode([base_text], convert_to_numpy=True)
    if base_vec is None or len(base_vec) == 0:
        raise ValueError("无法生成频道向量")
    
    base_tags = infer_topics_and_audience(base_vec)
    if not base_tags or not isinstance(base_tags, dict):
        base_info["topics"] = []
        base_info["audience"] = []
        base_tags = {"topics": [], "audience": []}
    else:
        base_info["topics"] = base_tags.get("topics", []) or []
        base_info["audience"] = base_tags.get("audience", []) or []
    
    # 边界检查：确保base_vec不为空
    if base_vec is None or len(base_vec) == 0:
        raise ValueError("无法生成频道向量")
    
    return base_vec[0], base_tags


def _collect_candidate_channels(
    base_vec: np.ndarray,
    base_info: Dict[str, Any],
    base_tags: Dict[str, Any],
    channel_id: str,
    report_progress: Callable[[float, str], None],
) -> List[str]:
    """
    收集候选频道ID（多种方式：本地索引、相关视频、标题搜索、主题/受众搜索）
    
    Args:
        base_vec: 基频道向量
        base_info: 基频道信息
        base_tags: 基频道标签
        channel_id: 基频道ID
        report_progress: 进度报告函数
    
    Returns:
        候选频道ID列表
    
    Raises:
        ValueError: 如果无法找到候选频道
    """
    candidate_ids: List[str] = []
    
    # 1. 从本地索引获取候选
    report_progress(25, "正在从本地索引搜索候选频道...")
    local_ids = get_candidates_from_local_index(
        base_vec, max_candidates=Config.CANDIDATE_COLLECTION["local_index_max"]
    )
    candidate_ids.extend(local_ids)
    
    # 2. 从相关视频获取候选
    recent_videos = []
    if base_info.get("recent_videos"):
        for v in base_info["recent_videos"]:
            if v and isinstance(v, dict):
                video_id = v.get("videoId")
                if video_id and isinstance(video_id, str):
                    recent_videos.append(video_id)
    
    if not recent_videos:
        raise ValueError(
            "该频道没有可用的公开视频，暂时无法分析相似频道。"
            "请确保频道有公开视频且未被限制访问。"
        )
    
    report_progress(30, "正在搜索相关视频频道...")
    try:
        related_ids = collect_candidate_channels_from_related_videos(
            recent_videos,
            per_video=Config.CANDIDATE_COLLECTION["related_videos_per_video"],
            limit=Config.CANDIDATE_COLLECTION["related_videos_limit"],
            use_for="search",
        )
        candidate_ids.extend(related_ids)
    except YouTubeQuotaExceededError:
        logger.warning("搜索相关视频频道时配额用尽，跳过此步骤")
        record_fallback_usage(
            fallback_type="skip_operation",
            endpoint="search",
            context="related_videos_search",
            success=True,
        )
    except (ValueError, Exception) as e:
        logger.debug(f"搜索相关视频频道失败: {e}")
    
    # 3. 按标题搜索
    report_progress(35, "正在按标题搜索相似频道...")
    try:
        title_based_ids = search_candidate_channels_by_title(
            base_info.get("title", ""), limit=Config.CANDIDATE_COLLECTION["title_search_limit"], use_for="search"
        )
        candidate_ids.extend(title_based_ids)
    except YouTubeQuotaExceededError:
        logger.warning("按标题搜索时配额用尽，跳过此步骤")
        record_fallback_usage(
            fallback_type="skip_operation",
            endpoint="search",
            context="title_search",
            success=True,
        )
    except Exception as e:
        logger.debug(f"按标题搜索失败: {e}")
    
    # 4. 按主题和受众搜索
    report_progress(40, "正在按主题和受众搜索相似频道...")
    
    def search_topic(topic: str) -> List[str]:
        """搜索单个主题"""
        try:
            return search_candidate_channels_by_title(
                topic, limit=Config.CANDIDATE_COLLECTION["topic_search_limit"], use_for="search"
            )
        except YouTubeQuotaExceededError:
            logger.debug(f"搜索主题 '{topic}' 时配额用尽")
            record_fallback_usage(
                fallback_type="skip_operation",
                endpoint="search",
                context=f"topic_search topic={topic}",
                success=True,
            )
            return []
        except Exception as e:
            logger.debug(f"搜索主题 '{topic}' 失败: {e}")
            return []
    
    def search_audience(audience: str) -> List[str]:
        """搜索单个受众"""
        try:
            search_query = f"{audience} crypto" if "crypto" not in audience.lower() else audience
            return search_candidate_channels_by_title(
                search_query, limit=Config.CANDIDATE_COLLECTION["audience_search_limit"], use_for="search"
            )
        except YouTubeQuotaExceededError:
            logger.debug(f"搜索受众 '{audience}' 时配额用尽")
            record_fallback_usage(
                fallback_type="skip_operation",
                endpoint="search",
                context=f"audience_search audience={audience}",
                success=True,
            )
            return []
        except Exception as e:
            logger.debug(f"搜索受众 '{audience}' 失败: {e}")
            return []
    
    # 并发执行主题和受众搜索
    topic_based_ids: List[str] = []
    audience_based_ids: List[str] = []
    
    search_workers = Config.get_thread_pool_size("search_workers", Config.CONCURRENT_PROCESSING["search_workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=search_workers) as executor:
        topic_futures = {
            executor.submit(search_topic, topic): topic 
            for topic in base_tags.get("topics", [])
        }
        audience_futures = {
            executor.submit(search_audience, audience): audience 
            for audience in base_tags.get("audience", [])
        }
        
        for future in concurrent.futures.as_completed(topic_futures):
            try:
                topic_ids = future.result()
                topic_based_ids.extend(topic_ids)
            except Exception as e:
                topic = topic_futures.get(future, "unknown")
                logger.debug(f"获取主题 '{topic}' 搜索结果时发生异常: {e}")
        
        for future in concurrent.futures.as_completed(audience_futures):
            try:
                aud_ids = future.result()
                audience_based_ids.extend(aud_ids)
            except Exception as e:
                audience = audience_futures.get(future, "unknown")
                logger.debug(f"获取受众 '{audience}' 搜索结果时发生异常: {e}")
    
    candidate_ids.extend(topic_based_ids)
    candidate_ids.extend(audience_based_ids)
    
    # 去重并限制数量
    seen_ids: set[str] = set()
    uniq_candidate_ids: List[str] = []
    for cid in candidate_ids:
        if cid not in seen_ids and cid != channel_id:
            seen_ids.add(cid)
            uniq_candidate_ids.append(cid)
    candidate_ids = uniq_candidate_ids[:Config.CANDIDATE_COLLECTION["max_candidates"]]
    
    if not candidate_ids:
        raise ValueError(
            "未能找到相似频道候选。"
            "这可能是因为：1) 频道内容过于独特；2) 网络连接问题；3) API配额用尽。"
            "请稍后重试或尝试使用其他频道。"
        )
    
    return candidate_ids


def _save_channels_to_db(
    base_info: Dict[str, Any],
    base_vec: np.ndarray,
    candidate_infos: List[Dict[str, Any]],
    cand_vecs_list: List[np.ndarray],
    tags_list: List[Dict[str, Any]],
) -> None:
    """
    将搜索过程中获取的频道信息保存到本地数据库。
    
    Args:
        base_info: 基频道信息
        base_vec: 基频道向量
        candidate_infos: 候选频道信息列表
        cand_vecs_list: 候选频道向量列表
        tags_list: 候选频道标签列表
    """
    if not DB_SAVE_AVAILABLE:
        return
    
    # 检查哪些频道需要保存（不在本地数据库中的，或需要更新的）
    from database import get_db_connection
    import sqlite3
    
    # 收集所有频道ID
    all_channel_ids: List[str] = []
    base_cid = base_info.get("channelId") if base_info else None
    if base_cid and isinstance(base_cid, str):
        all_channel_ids.append(base_cid)
    
    if candidate_infos:
        for info in candidate_infos:
            if info and isinstance(info, dict):
                cid = info.get("channelId")
                if cid and isinstance(cid, str) and cid not in all_channel_ids:
                    all_channel_ids.append(cid)
    
    # 检查哪些频道已经在数据库中
    existing_channels = set()
    if all_channel_ids:  # 只有在有频道ID时才查询数据库
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                placeholders = ",".join(["?" for _ in all_channel_ids])
                cur.execute(
                    f"SELECT channel_id FROM channels WHERE channel_id IN ({placeholders})",
                    all_channel_ids
                )
                rows = cur.fetchall()
                existing_channels = {row[0] for row in rows if row and len(row) > 0}
        except Exception as e:
            logger.debug(f"检查现有频道时出错: {e}")
            # 如果检查失败，尝试保存所有频道（数据库会自动处理冲突）
            existing_channels = set()
    
    # 准备要保存的频道数据
    channels_to_save: List[Tuple] = []
    
    # 保存基频道
    base_cid = base_info.get("channelId")
    if base_cid:
        # 即使存在也保存（更新数据）
        base_data = _prepare_channel_data_for_db(base_info, base_vec)
        if base_data:
            channels_to_save.append(base_data)
    
    # 保存候选频道（只保存不在数据库中的，或需要更新的）
    for idx, info in enumerate(candidate_infos):
        cid = info.get("channelId")
        if not cid:
            continue
        
        # 如果频道不在数据库中，则保存（数据库的 ON CONFLICT 会处理已存在频道的更新）
        # 对于已存在的频道，也保存以更新数据（但可以优化为只更新过期的）
        if cid not in existing_channels:
            # 获取对应的向量和标签
            vec = None
            if idx < len(cand_vecs_list):
                vec = cand_vecs_list[idx]
            
            tags = {}
            if idx < len(tags_list):
                tags = tags_list[idx]
            
            # 合并标签到频道信息中
            info_with_tags = dict(info)
            if tags:
                info_with_tags["topics"] = tags.get("topics", info_with_tags.get("topics", []))
                info_with_tags["audience"] = tags.get("audience", info_with_tags.get("audience", []))
            
            # 如果没有向量，尝试从已有数据构建
            if vec is None:
                try:
                    text = build_text_for_channel(info_with_tags)
                    model = get_embed_model()
                    vec = model.encode([text], convert_to_numpy=True)[0]
                except Exception as e:
                    logger.debug(f"无法为频道 {cid} 生成向量: {e}")
                    continue
            
            channel_data = _prepare_channel_data_for_db(info_with_tags, vec)
            if channel_data:
                channels_to_save.append(channel_data)
    
    # 批量保存到数据库
    if channels_to_save:
        try:
            # _batch_upsert_channels 内部已经会失效缓存，这里不需要重复失效
            _batch_upsert_channels(channels_to_save)
            logger.info(f"成功保存 {len(channels_to_save)} 个频道到本地数据库（缓存已自动失效）")
        except Exception as e:
            logger.warning(f"批量保存频道到数据库失败: {e}", exc_info=True)


def _prepare_channel_data_for_db(info: Dict[str, Any], vec: np.ndarray) -> Tuple | None:
    """
    将频道信息转换为数据库格式，包含数据验证。
    
    Args:
        info: 频道信息字典
        vec: 频道向量
    
    Returns:
        数据库格式的元组，如果转换失败则返回 None
    """
    try:
        cid = info.get("channelId")
        if not cid or not isinstance(cid, str) or not cid.strip():
            return None
        
        if vec is None:
            return None
        
        # 验证和清理数值字段
        try:
            sub_count = int(info.get("subscriberCount") or 0)
            view_count = int(info.get("viewCount") or 0)
            # 确保非负
            sub_count = max(0, sub_count)
            view_count = max(0, view_count)
        except (ValueError, TypeError) as e:
            logger.debug(f"验证频道数值字段失败: {e}，使用默认值0")
            sub_count = 0
            view_count = 0
        
        # 清理字符串字段
        title = (info.get("title") or "").strip()[:500]  # 限制长度
        description = (info.get("description") or "").strip()[:10000]  # 限制长度
        
        # 验证JSON字段
        emails = info.get("emails", [])
        topics = info.get("topics", [])
        audience = info.get("audience", [])
        
        if not isinstance(emails, list):
            emails = []
        if not isinstance(topics, list):
            topics = []
        if not isinstance(audience, list):
            audience = []
        
        # 准备数据格式：(channel_id, title, description, subscriber_count, view_count,
        #                country, language, emails_json, topics_json, audience_json, embedding_bytes)
        return (
            cid.strip(),
            title,
            description,
            sub_count,
            view_count,
            info.get("country"),
            info.get("defaultLanguage") or info.get("defaultAudioLanguage"),
            json.dumps(emails, ensure_ascii=False),
            json.dumps(topics, ensure_ascii=False),
            json.dumps(audience, ensure_ascii=False),
            vec.astype(np.float32).tobytes(),
        )
    except Exception as e:
        logger.debug(f"准备频道数据时出错: {e}")
        return None


async def get_similar_channels_by_url(
    channel_url: str,
    max_results: int = 20,
    min_subscribers: int | None = None,
    max_subscribers: int | None = None,
    preferred_language: str | None = None,
    preferred_region: str | None = None,
    min_similarity: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    bd_mode: bool = False,
) -> Dict[str, Any]:
    """
    高层封装：给定频道链接，返回相似频道列表（包含相似度）。
    
    此函数已重构（CP-y6-03：函数职责单一），将复杂逻辑拆分为多个职责单一的辅助函数。
    
    Args:
        channel_url: YouTube 频道链接
        max_results: 最大返回结果数（1-200）
        min_subscribers: 最小订阅数筛选（不能为负数）
        max_subscribers: 最大订阅数筛选（不能为负数）
        preferred_language: 偏好语言（如 "en", "zh-Hans"）
        preferred_region: 偏好地区（如 "US", "CN"）
        min_similarity: 最低语义相似度（0-1）
        progress_callback: 可选的回调函数，接收 (progress: float, message: str) 参数
        bd_mode: 是否启用BD模式（交易所BD寻找KOL专用）
    
    Returns:
        包含 base_channel 和 similar_channels 的字典
        BD模式下还包含 bd_metrics 相关字段
    
    Raises:
        ValueError: 如果输入参数无效
    """
    # 参数验证
    channel_url = _validate_search_params(
        channel_url, max_results, min_subscribers, max_subscribers, min_similarity
    )
    
    def _report_progress(progress: float, message: str) -> None:
        """报告进度（内部辅助函数）"""
        if progress_callback:
            progress_callback(progress, message)
    
    quota_exceeded_channels: List[Dict[str, str]] = []
    
    # 1. 解析频道链接
    _report_progress(5, "正在解析频道链接...")
    channel_id = extract_channel_id_from_url(channel_url, use_for="search")
    
    # 2. 获取并丰富基频道信息
    base_info = _get_base_channel_info(channel_id, _report_progress)
    _enrich_base_channel_info(base_info, channel_id, _report_progress)
    
    # 3. 计算基频道向量和标签
    base_vec, base_tags = _compute_base_channel_embedding(base_info, _report_progress)

    # 4. 收集候选频道
    candidate_ids = _collect_candidate_channels(
        base_vec, base_info, base_tags, channel_id, _report_progress
    )

    # 提前计算用于粗筛的变量
    # 如果用户没有指定 min_subscribers，默认使用 1000
    if min_subscribers is None:
        min_subs_filter = 1000
    else:
        min_subs_filter = None if min_subscribers == 0 else min_subscribers
    max_subs_filter = None if max_subscribers == 0 else max_subscribers
    base_subs_for_filter = base_info.get("subscriberCount") or 0
    preferred_language_norm = (preferred_language or "").lower()
    preferred_region_norm = (preferred_region or "").upper()

    _report_progress(50, f"正在从本地数据库获取 {len(candidate_ids)} 个候选频道信息...")
    # 3. 优化：优先从本地数据库获取频道信息，进行粗筛，只对缺失的调用API
    # 从本地数据库获取频道信息，如果数据过期则自动在后台更新（CP-y4-02）
    local_infos = get_channel_info_from_local_db(
        candidate_ids, 
        max_age_days=7,
        auto_update_expired=True  # 启用自动更新过期数据
    )
    
    # 粗筛：使用本地数据库信息进行初步筛选（如果可用）
    # 这样可以在调用API前过滤掉明显不符合条件的候选
    if local_infos and (min_subs_filter is not None or max_subs_filter is not None or 
                        preferred_language_norm or preferred_region_norm):
        filtered_local_infos = []
        for info in local_infos:
            cid = info.get("channelId")
            subs = info.get("subscriberCount", 0)
            
            # 订阅数筛选
            if min_subs_filter is not None and subs < min_subs_filter:
                continue
            if max_subs_filter is not None and subs > max_subs_filter:
                continue
            if min_subs_filter is None and base_subs_for_filter > 0:
                dynamic_min = max(
                    Config.SUBSCRIBER_FILTER["min_absolute"],
                    int(base_subs_for_filter * Config.SUBSCRIBER_FILTER["min_dynamic_ratio"])
                )
                if subs < dynamic_min:
                    continue
            
            # 语言/地区筛选
            if preferred_language_norm:
                lang = (info.get("defaultLanguage") or "").lower()
                if lang and preferred_language_norm not in lang:
                    continue
            if preferred_region_norm:
                country = (info.get("country") or "").upper()
                if country and country != preferred_region_norm:
                    continue
            
            filtered_local_infos.append(info)
        
        local_infos = filtered_local_infos
    
    local_info_map = {info["channelId"]: info for info in local_infos}
    local_channel_ids = set(local_info_map.keys())
    
    # 找出需要从API获取的频道ID
    missing_ids = [cid for cid in candidate_ids if cid not in local_channel_ids]
    
    candidate_infos: List[Dict[str, Any]] = []
    if missing_ids:
        _report_progress(52, f"正在从API获取 {len(missing_ids)} 个缺失频道的信息...")
        try:
            api_infos = batch_get_channels_info(missing_ids, use_for="search")
            candidate_infos.extend(api_infos)
        except YouTubeQuotaExceededError:
            logger.warning(f"批量获取频道信息时配额用尽，记录 {len(missing_ids)} 个频道的链接")
            # 记录配额不足的频道链接
            for cid in missing_ids:
                quota_exceeded_channels.append({
                    "channelId": cid,
                    "channelUrl": f"https://www.youtube.com/channel/{cid}"
                })
            # 如果配额用尽，只使用本地数据库中的频道
            if local_infos:
                # 记录降级统计（成功使用本地数据）
                record_fallback_usage(
                    fallback_type="local_db_batch_channels",
                    endpoint="channels",
                    context=f"batch_get {len(missing_ids)} missing, {len(local_infos)} from local",
                    success=True,
                )
            else:
                # 记录降级失败
                record_fallback_usage(
                    fallback_type="local_db_batch_channels",
                    endpoint="channels",
                    context=f"batch_get {len(missing_ids)} missing, no local data",
                    success=False,
                )
                raise ValueError(
                    "API配额已用尽，且本地数据库中没有足够的频道信息。\n"
                    "解决方案：\n"
                    "1. 请等待配额重置（通常在每天UTC 00:00重置）\n"
                    "2. 使用 build_channel_index.py 预先建立频道索引，以提升本地数据覆盖率\n"
                    "3. 或申请更高的API配额"
                )
    
    # 添加本地数据库中的频道信息
    candidate_infos.extend(local_infos)
    
    # 按原始顺序重新排序（保持优先级）
    candidate_id_map = {info["channelId"]: info for info in candidate_infos}
    candidate_infos = [candidate_id_map[cid] for cid in candidate_ids if cid in candidate_id_map]
    
    if not candidate_infos:
        raise ValueError("经过粗筛后没有符合条件的候选频道，请调整筛选条件后重试。")

    _report_progress(60, "正在并行获取候选频道的最近视频...")
    # 优化：并行获取最近视频，提高效率
    
    def fetch_videos_and_emails(info: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]], List[str], str | None]:
        """
        获取单个频道的最近视频和邮箱
        
        Returns:
            (channel_id, recent_videos, emails, error_message) 元组
        """
        cid = info.get("channelId", "")
        try:
            recent_v = get_recent_video_snippets_for_channel(
                cid, max_results=Config.CHANNEL_INFO["recent_videos_for_similarity"], use_for="search"
            )
            # 提取邮箱：从频道描述和视频描述中提取
            emails: List[str] = []
            emails.extend(extract_emails_from_text(info.get("description", "")))
            for v in recent_v:
                emails.extend(extract_emails_from_text(v.get("description", "")))
            return (cid, recent_v, list(dict.fromkeys(emails)), None)
        except YouTubeQuotaExceededError:
            logger.warning(f"获取频道 {cid} 的视频信息时配额用尽")
            # 记录配额不足的频道（如果还没有记录）
            if not any(ch["channelId"] == cid for ch in quota_exceeded_channels):
                quota_exceeded_channels.append({
                    "channelId": cid,
                    "channelUrl": f"https://www.youtube.com/channel/{cid}"
                })
            # 记录降级统计（跳过操作）
            record_fallback_usage(
                fallback_type="skip_operation",
                endpoint="playlistItems",
                context=f"candidate_videos channel_id={cid}",
                success=True,
            )
            return (cid, [], [], "配额已用尽")
        except Exception as e:
            logger.debug(f"获取频道 {cid} 的视频信息失败: {e}", exc_info=True)
            return (cid, [], [], str(e))
    
    # 并行获取最近视频
    total_candidates = len(candidate_infos)
    videos_and_emails_map: Dict[str, tuple[List[Dict[str, Any]], List[str]]] = {}
    failed_channels: List[Dict[str, str]] = []  # 记录失败的频道及其原因
    
    # 使用配置化的线程池大小
    video_fetch_workers = Config.get_thread_pool_size("video_fetch_workers", Config.CONCURRENT_PROCESSING["video_fetch_workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=video_fetch_workers) as executor:
        future_to_cid = {
            executor.submit(fetch_videos_and_emails, info): info.get("channelId", "")
            for info in candidate_infos
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_cid):
            completed += 1
            if completed % 20 == 0:
                _report_progress(60 + int(10 * completed / total_candidates), f"正在处理候选频道 {completed}/{total_candidates}...")
            try:
                result = future.result()
                if not result or len(result) != 4:
                    cid = future_to_cid.get(future, "unknown")
                    failed_channels.append({
                        "channelId": cid,
                        "channelUrl": f"https://www.youtube.com/channel/{cid}" if cid != "unknown" else "",
                        "reason": "返回格式无效"
                    })
                    logger.warning(f"获取频道信息返回格式无效: {result}")
                    continue
                
                cid, recent_v, emails, error_msg = result
                if cid and isinstance(cid, str):
                    if error_msg:
                        failed_channels.append({
                            "channelId": cid,
                            "channelUrl": f"https://www.youtube.com/channel/{cid}",
                            "reason": error_msg
                        })
                        logger.debug(f"频道 {cid} 的视频获取失败: {error_msg}")
                    # 确保recent_v和emails不为None
                    videos_and_emails_map[cid] = (
                        recent_v if recent_v is not None else [],
                        emails if emails is not None else []
                    )
            except Exception as e:
                cid = future_to_cid.get(future, "unknown")
                failed_channels.append({
                    "channelId": cid,
                    "channelUrl": f"https://www.youtube.com/channel/{cid}" if cid != "unknown" else "",
                    "reason": f"异常: {type(e).__name__}: {str(e)}"
                })
                logger.warning(f"获取频道 {cid} 信息时发生异常: {e}", exc_info=True)
    
    # 记录失败的频道数量（如果较多）
    if len(failed_channels) > 0:
        logger.info(f"共有 {len(failed_channels)} 个频道的视频信息获取失败")
        # 去重：如果某个频道已经在quota_exceeded_channels中，就不在failed_channels中重复记录
        quota_exceeded_ids = {ch["channelId"] for ch in quota_exceeded_channels}
        failed_channels = [ch for ch in failed_channels if ch["channelId"] not in quota_exceeded_ids]
    
    # 合并视频和邮箱信息到频道信息中
    for info in candidate_infos:
        if not info or not isinstance(info, dict):
            continue
        
        cid = info.get("channelId")
        if not cid:
            continue
        
        if cid in videos_and_emails_map:
            recent_v, emails = videos_and_emails_map[cid]
            info["recent_videos"] = recent_v if recent_v is not None else []
            # 合并邮箱（如果频道描述中有邮箱，也要加上）
            existing_emails = info.get("emails")
            if not existing_emails:
                existing_emails = []
            if not isinstance(existing_emails, list):
                existing_emails = []
            
            emails_to_add = emails if emails is not None else []
            if not isinstance(emails_to_add, list):
                emails_to_add = []
            
            all_emails = list(dict.fromkeys(existing_emails + emails_to_add))
            info["emails"] = all_emails
        else:
            info["recent_videos"] = []
            if "emails" not in info or info.get("emails") is None:
                info["emails"] = []

    _report_progress(70, "正在计算相似度...")
    # 4. 优化：优先复用本地数据库中的向量，只计算新频道的向量
    local_embeddings = get_embeddings_from_local_db([info["channelId"] for info in candidate_infos])
    
    # 分离需要计算向量的频道和已有向量的频道
    texts_to_encode: List[str] = []
    indices_to_encode: List[int] = []
    
    for idx, info in enumerate(candidate_infos):
        cid = info.get("channelId")
        if cid not in local_embeddings:
            texts_to_encode.append(build_text_for_channel(info))
            indices_to_encode.append(idx)
    
    # 批量计算新频道的向量（分批处理以避免内存不足）
    new_vecs: Dict[int, np.ndarray] = {}
    if texts_to_encode:
        # 使用配置的批量大小，避免一次性处理太多导致内存不足
        embedding_batch_size = Config.get_config_value("embedding.batch_size", Config.EMBEDDING_BATCH_SIZE)
        # 如果批量大小太大，限制为更小的值（内存优化）
        embedding_batch_size = min(embedding_batch_size, 16)  # 最多16个一批，减少内存占用
        
        for batch_start in range(0, len(texts_to_encode), embedding_batch_size):
            batch_end = min(batch_start + embedding_batch_size, len(texts_to_encode))
            batch_texts = texts_to_encode[batch_start:batch_end]
            batch_indices = indices_to_encode[batch_start:batch_end]
            
            try:
                encoded_vecs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                for i, idx in enumerate(batch_indices):
                    new_vecs[idx] = encoded_vecs[i]
            except MemoryError:
                # 如果内存不足，尝试更小的批量
                logger.warning(f"批量编码时内存不足，尝试更小的批量大小")
                smaller_batch = max(1, embedding_batch_size // 2)
                for j, idx in enumerate(batch_indices):
                    try:
                        text = batch_texts[j]
                        vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
                        new_vecs[idx] = vec
                    except Exception as e:
                        logger.warning(f"为频道索引 {idx} 编码向量失败: {e}")
                        continue
    
    # 构建完整的向量列表
    cand_vecs_list: List[np.ndarray] = []
    for idx, info in enumerate(candidate_infos):
        cid = info.get("channelId")
        if cid in local_embeddings:
            cand_vecs_list.append(local_embeddings[cid])
        elif idx in new_vecs:
            cand_vecs_list.append(new_vecs[idx])
        else:
            # 兜底：如果都没有，重新计算
            text = build_text_for_channel(info)
            vec = model.encode([text], convert_to_numpy=True)[0]
            cand_vecs_list.append(vec)
    
    # 转换为numpy数组，保持与原有代码兼容
    if cand_vecs_list:
        cand_vecs = np.vstack(cand_vecs_list)
    else:
        cand_vecs = np.array([])

    # 批量推理标签（优化：一次性处理所有候选频道）
    _report_progress(75, "正在批量推理候选频道的标签...")
    if cand_vecs.size > 0:
        batch_tags = infer_topics_and_audience(cand_vecs)
        # 如果返回的是列表，说明是批量结果；否则是单个结果（向后兼容）
        if isinstance(batch_tags, list):
            tags_list = batch_tags
        else:
            tags_list = [batch_tags]
    else:
        tags_list = []

    # 计算相似度，并按规则过滤 + 计算总评分
    scored: List[Tuple[float, Dict[str, Any]]] = []

    base_subs = base_info.get("subscriberCount") or 0
    base_topics = set(base_info.get("topics") or [])
    base_audience = set(base_info.get("audience") or [])

    for idx, (vec, info) in enumerate(zip(cand_vecs, candidate_infos)):
        info_with_score = dict(info)

        # 语义相似度
        sim = cosine_similarity(base_vec, vec)
        if min_similarity is not None and sim < min_similarity:
            continue

        # 使用批量推理的标签结果
        if idx < len(tags_list):
            tags = tags_list[idx]
            info_with_score["topics"] = tags["topics"]
            info_with_score["audience"] = tags["audience"]
        else:
            # 兜底：如果批量推理失败，使用空标签
            info_with_score["topics"] = []
            info_with_score["audience"] = []

        # Topics / Audience 标签重合度
        cand_topics = set(info_with_score["topics"])
        cand_aud = set(info_with_score["audience"])
        topic_overlap = calculate_tag_overlap(base_topics, cand_topics)
        aud_overlap = calculate_tag_overlap(base_audience, cand_aud)
        tag_score = 0.5 * topic_overlap + 0.5 * aud_overlap

        subs = int(info_with_score.get("subscriberCount") or 0)
        scale = scale_score(base_subs, subs)
        # 综合得分
        total = calculate_total_score(sim, tag_score, scale)

        info_with_score["similarity"] = sim
        info_with_score["scale_score"] = scale
        info_with_score["tags_score"] = tag_score
        info_with_score["total_score"] = total

        # BD模式：计算BD专属评分
        if bd_mode:
            bd_result = calculate_full_bd_metrics(
                channel_info=info_with_score,
                semantic_sim=sim,
                scale_score_val=scale,
            )
            info_with_score["bd_total_score"] = bd_result["bd_total_score"]
            info_with_score["bd_priority"] = bd_result["bd_priority"]
            info_with_score["bd_metrics"] = bd_result["bd_metrics"]
            info_with_score["bd_breakdown"] = bd_result["bd_breakdown"]
            info_with_score["bd_recommendation"] = bd_result["bd_recommendation"]
            # BD模式下使用BD总分作为排序依据
            total = bd_result["bd_total_score"]

        scored.append((total, info_with_score))

    _report_progress(85, "正在排序和筛选结果...")
    # 按综合评分排序
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [item for _, item in scored[:max_results]]
    
    _report_progress(90, "正在计算互动率指标...")
    # 为最终返回的 top 频道计算 E.R. 和 V.R.
    for channel_info in top:
        cid = channel_info.get("channelId")
        subs = channel_info.get("subscriberCount", 0)
        if not cid or subs <= 0:
            channel_info["avg_views"] = 0.0
            channel_info["avg_likes"] = 0.0
            channel_info["engagement_rate"] = 0.0
            channel_info["view_rate"] = 0.0
            continue
        
        try:
            # 获取最近视频的平均统计数据
            stats = get_recent_videos_stats(cid, max_results=Config.CHANNEL_INFO["stats_videos_count"], use_for="search")
            avg_views = stats["avg_views"]
            avg_likes = stats["avg_likes"]
            
            # 计算 E.R. 和 V.R.
            # subs > 0 已在外层检查，这里直接计算
            engagement_rate = (avg_likes / subs * 100)
            view_rate = (avg_views / subs * 100)
            
            channel_info["avg_views"] = round(avg_views, 1)
            channel_info["avg_likes"] = round(avg_likes, 1)
            channel_info["engagement_rate"] = round(engagement_rate, 1)
            channel_info["view_rate"] = round(view_rate, 1)
        except Exception:
            # 如果获取统计数据失败，设置为 0
            channel_info["avg_views"] = 0.0
            channel_info["avg_likes"] = 0.0
            channel_info["engagement_rate"] = 0.0
            channel_info["view_rate"] = 0.0
    
    # 也为基频道计算 E.R. 和 V.R.
    # base_subs 已在前面计算过，直接使用
    if base_subs > 0:
        try:
            base_stats = get_recent_videos_stats(channel_id, max_results=Config.CHANNEL_INFO["stats_videos_count"], use_for="search")
            base_avg_views = base_stats["avg_views"]
            base_avg_likes = base_stats["avg_likes"]
            base_info["avg_views"] = round(base_avg_views, 1)
            base_info["avg_likes"] = round(base_avg_likes, 1)
            # base_subs > 0 已在外层检查，这里直接计算
            base_info["engagement_rate"] = round((base_avg_likes / base_subs * 100), 1)
            base_info["view_rate"] = round((base_avg_views / base_subs * 100), 1)
        except Exception:
            base_info["avg_views"] = 0.0
            base_info["avg_likes"] = 0.0
            base_info["engagement_rate"] = 0.0
            base_info["view_rate"] = 0.0
    
    _report_progress(95, "正在保存新频道到本地数据库...")
    # 5. 自动保存新发现的频道到本地数据库
    if DB_SAVE_AVAILABLE:
        try:
            _save_channels_to_db(base_info, base_vec, candidate_infos, cand_vecs_list, tags_list)
        except Exception as e:
            # 保存失败不影响搜索结果返回
            logger.warning(f"保存频道到数据库时发生错误（不影响搜索结果）: {e}", exc_info=True)
    
    # BD模式：为基频道也计算BD指标
    if bd_mode:
        base_bd_result = calculate_full_bd_metrics(
            channel_info=base_info,
            semantic_sim=1.0,  # 自身相似度为1
            scale_score_val=1.0,  # 自身规模得分为1
        )
        base_info["bd_metrics"] = base_bd_result["bd_metrics"]
        base_info["bd_breakdown"] = base_bd_result["bd_breakdown"]
    
    _report_progress(100, "完成！")
    
    result = {
        "base_channel": base_info,
        "similar_channels": top,
        "quota_exceeded_channels": quota_exceeded_channels,  # 配额不足的频道链接列表
        "failed_channels": failed_channels,  # 其他原因失败的频道列表（包含失败原因）
    }
    
    # BD模式：添加额外的统计信息
    if bd_mode:
        high_priority_count = sum(1 for ch in top if ch.get("bd_priority") == "high")
        medium_priority_count = sum(1 for ch in top if ch.get("bd_priority") == "medium")
        low_priority_count = sum(1 for ch in top if ch.get("bd_priority") == "low")
        with_email_count = sum(1 for ch in top if ch.get("bd_metrics", {}).get("commercialization", {}).get("has_email", False))
        with_competitor_count = sum(1 for ch in top if ch.get("bd_metrics", {}).get("competitor_detection", {}).get("has_competitor_collab", False))
        
        result["bd_summary"] = {
            "mode": "bd",
            "total_results": len(top),
            "high_priority": high_priority_count,
            "medium_priority": medium_priority_count,
            "low_priority": low_priority_count,
            "with_email": with_email_count,
            "with_competitor_collab": with_competitor_count,
            "recommendation": f"共找到 {high_priority_count} 个高优先级KOL，{with_email_count} 个有联系邮箱",
        }
    
    return result
