import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
import threading

# 添加项目根目录到 Python 路径（修复模块导入问题）
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock
from typing import Iterable, List, Tuple, Optional, Dict, Any

# 强制刷新输出，避免缓冲导致看不到实时日志
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 添加导入进度提示（numpy导入很慢，需要20-30秒）
print("正在加载依赖库（numpy可能需要20-30秒，请稍候）...", flush=True)
import numpy as np
print("依赖库加载完成", flush=True)

# 分步导入项目模块，添加进度提示
print("正在导入项目模块...", flush=True)
print("  - 导入 core.candidate_collection...", flush=True)
from core.candidate_collection import search_candidate_channels_by_title
print("  - 导入 infrastructure.cache...", flush=True)
from infrastructure.cache import invalidate_all_channel_caches
print("  - 导入 core.channel_info...", flush=True)
from core.channel_info import (
    get_channel_basic_info,
    get_recent_video_snippets_for_channel,
)
print("  - 导入 core.channel_parser...", flush=True)
from core.channel_parser import extract_channel_id_from_url
print("  - 导入 infrastructure.config...", flush=True)
from infrastructure.config import Config
print("    infrastructure.config 导入完成", flush=True)
print("  - 导入 core.embedding（sentence_transformers可能需要10-20秒）...", flush=True)
from core.embedding import (
    get_embed_model,
    infer_topics_and_audience,
    ensure_label_embeddings,
    encode_async,
)
print("    core.embedding 导入完成", flush=True)
print("  - 导入 infrastructure.logger...", flush=True)
from infrastructure.logger import get_logger
print("  - 导入 infrastructure.utils...", flush=True)
from infrastructure.utils import build_text_for_channel, extract_emails_from_text
print("  - 导入 core.youtube_api...", flush=True)
from core.youtube_api import YouTubeQuotaExceededError, yt_get
print("项目模块导入完成", flush=True)

logger = get_logger()


DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "channel_index.db"))
CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "discovered_channels_cache.json"))

# 线程安全的锁，用于数据库操作
_db_lock = Lock()


def _get_keywords_hash(keywords: Iterable[str] | None) -> str:
    """
    生成关键词列表的哈希值，用于判断关键词是否变化。
    
    Args:
        keywords: 关键词列表
    
    Returns:
        关键词的MD5哈希值
    """
    if not keywords:
        return ""
    # 排序后生成哈希，确保顺序不影响结果
    sorted_keywords = sorted(set(keywords))
    keywords_str = "|".join(sorted_keywords)
    return hashlib.md5(keywords_str.encode('utf-8')).hexdigest()


def _load_cached_channel_ids() -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    从缓存文件加载已发现的频道ID列表。
    
    Returns:
        (channel_ids, keywords_hash, last_updated) - 如果缓存不存在，返回 ([], None, None)
    """
    if not os.path.exists(CACHE_PATH):
        return ([], None, None)
    
    try:
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            channel_ids = cache_data.get("channels", [])
            keywords_hash = cache_data.get("keywords_hash")
            last_updated = cache_data.get("last_updated")
            logger.info(f"从缓存加载了 {len(channel_ids)} 个已发现的频道ID")
            return (channel_ids, keywords_hash, last_updated)
    except Exception as e:
        logger.warning(f"加载频道ID缓存失败: {e}，将重新搜索")
        return ([], None, None)


def _save_cached_channel_ids(channel_ids: List[str], keywords_hash: str) -> None:
    """
    将发现的频道ID列表保存到缓存文件。
    
    Args:
        channel_ids: 频道ID列表
        keywords_hash: 关键词哈希值
    """
    try:
        cache_data = {
            "channels": list(set(channel_ids)),  # 去重
            "keywords_hash": keywords_hash,
            "last_updated": datetime.now().isoformat(),
        }
        # 确保目录存在
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(cache_data['channels'])} 个频道ID到缓存文件")
    except Exception as e:
        logger.warning(f"保存频道ID缓存失败: {e}")


def _should_refresh_cache(cached_hash: Optional[str], current_hash: str) -> bool:
    """
    判断是否需要刷新缓存（重新搜索关键词）。
    
    Args:
        cached_hash: 缓存中的关键词哈希
        current_hash: 当前关键词哈希
    
    Returns:
        如果需要刷新返回True，否则返回False
    """
    if not cached_hash:
        return True  # 没有缓存，需要搜索
    if cached_hash != current_hash:
        logger.info("关键词列表已变化，需要刷新缓存")
        return True  # 关键词变化了，需要重新搜索
    return False  # 关键词未变化，可以使用缓存


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """
    确保数据库表结构存在，并创建必要的索引以提高查询性能。
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY NOT NULL,
            title TEXT,
            description TEXT,
            subscriber_count INTEGER DEFAULT 0 CHECK(subscriber_count >= 0),
            view_count INTEGER DEFAULT 0 CHECK(view_count >= 0),
            country TEXT,
            language TEXT,
            emails TEXT,
            topics TEXT,
            audience TEXT,
            recent_videos TEXT,
            engagement_rate REAL DEFAULT 0.0 CHECK(engagement_rate >= 0.0),
            view_rate REAL DEFAULT 0.0 CHECK(view_rate >= 0.0),
            competitor_detection TEXT,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_embeddings (
            channel_id TEXT PRIMARY KEY NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    
    # 为现有数据库添加新字段（向后兼容）
    # 检查字段是否存在，如果不存在则添加
    cur.execute("PRAGMA table_info(channels)")
    existing_columns = [row[1] for row in cur.fetchall()]
    
    if "recent_videos" not in existing_columns:
        cur.execute("ALTER TABLE channels ADD COLUMN recent_videos TEXT")
        logger.debug("已添加 recent_videos 字段到 channels 表")
    
    if "engagement_rate" not in existing_columns:
        cur.execute("ALTER TABLE channels ADD COLUMN engagement_rate REAL DEFAULT 0.0 CHECK(engagement_rate >= 0.0)")
        logger.debug("已添加 engagement_rate 字段到 channels 表")
    
    if "view_rate" not in existing_columns:
        cur.execute("ALTER TABLE channels ADD COLUMN view_rate REAL DEFAULT 0.0 CHECK(view_rate >= 0.0)")
        logger.debug("已添加 view_rate 字段到 channels 表")
    
    if "competitor_detection" not in existing_columns:
        cur.execute("ALTER TABLE channels ADD COLUMN competitor_detection TEXT")
        logger.debug("已添加 competitor_detection 字段到 channels 表")
    
    # 创建索引以提高查询性能
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_channels_updated_at 
        ON channels(updated_at)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_channels_subscriber_count 
        ON channels(subscriber_count)
        """
    )
    
    conn.commit()
    logger.debug("数据库表结构和索引已确保存在")


async def _discover_channels_by_keyword(keyword: str, max_results: int = 20) -> List[str]:
    """
    通过关键词搜索发现频道 ID（只搜索一次）。
    默认 max_results=20 以进一步减少配额消耗。
    """
    try:
        ids = await search_candidate_channels_by_title(keyword, limit=max_results, use_for="index")
        return ids
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"关键词 '{keyword}' 搜索失败（网络错误）: {e}")
        return []
    except Exception as e:
        logger.warning(f"关键词 '{keyword}' 搜索失败: {e}")
        return []


def _channel_needs_update(conn: sqlite3.Connection, channel_id: str, max_age_days: int = 60) -> bool:
    """
    检查频道是否需要更新。
    如果频道不存在或数据超过 max_age_days 天，返回 True。
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT updated_at FROM channels WHERE channel_id = ?
        """,
        (channel_id,),
    )
    row = cur.fetchone()
    if not row:
        return True  # 频道不存在，需要获取
    
    updated_at_str = row[0]
    if not updated_at_str:
        return True
    
    # 解析时间戳
    try:
        # SQLite 的 datetime 格式可能是 "YYYY-MM-DD HH:MM:SS"
        if "T" in updated_at_str:
            # ISO 格式
            updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            if updated_at.tzinfo:
                updated_at = updated_at.replace(tzinfo=None)
        else:
            # SQLite 格式
            updated_at = datetime.strptime(updated_at_str, "%Y-%m-%d %H:%M:%S")
        
        age = datetime.now() - updated_at
        return age.days > max_age_days
    except Exception as e:
        logger.debug(f"解析频道 {channel_id} 的更新时间失败: {e}，保守起见选择更新")
        return True  # 解析失败，保守起见选择更新


async def _process_channel_data(channel_id: str, recent_videos_count: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    处理单个频道：获取信息、计算向量和标签。
    这是公共逻辑，不涉及数据库操作。
    
    Args:
        channel_id: 频道 ID
        recent_videos_count: 获取最近多少个视频
    
    Returns:
        (info_dict, embedding_vector) - 如果成功，返回 (info, vec)，否则返回 (None, None)
        
    Raises:
        YouTubeQuotaExceededError: 如果 API 配额已用完
    """
    # #region agent log
    try:
        import json
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_data:ENTRY","message":"开始处理频道数据","data":{"channel_id":channel_id,"recent_videos_count":recent_videos_count},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    try:
        info = await get_channel_basic_info(channel_id, use_for="index")
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_data:AFTER_GET_INFO","message":"成功获取频道基础信息","data":{"channel_id":channel_id,"has_title":bool(info.get("title")),"has_description":bool(info.get("description"))},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
    except YouTubeQuotaExceededError as e:
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_data:QUOTA_ERROR","message":"API配额已用完","data":{"channel_id":channel_id,"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        logger.error(f"YouTube API 配额已用完: {e}")
        raise
    except Exception as e:
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_data:GET_INFO_ERROR","message":"获取频道信息失败","data":{"channel_id":channel_id,"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        logger.warning(f"获取频道 {channel_id} 失败: {e}")
        return (None, None)

    # 获取最近视频（用于更精准的 embedding）
    try:
        recent_videos = await get_recent_video_snippets_for_channel(channel_id, max_results=recent_videos_count, use_for="index")
    except YouTubeQuotaExceededError as e:
        logger.error(f"YouTube API 配额已用完: {e}")
        raise
    except Exception as e:
        logger.warning(f"获取频道 {channel_id} 的最近视频失败: {e}")
        recent_videos = []
    info["recent_videos"] = recent_videos

    # 提取邮箱
    emails: List[str] = []
    emails.extend(extract_emails_from_text(info.get("description", "")))
    for v in recent_videos:
        emails.extend(extract_emails_from_text(v.get("description", "")))
    info["emails"] = list(dict.fromkeys(emails))

    # 计算向量和标签
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"build_channel_index.py:_process_channel_data:BEFORE_EMBEDDING","message":"准备计算向量","data":{"channel_id":channel_id,"text_length":len(build_text_for_channel(info))},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    model = get_embed_model()
    ensure_label_embeddings(model)  # 初始化标签向量（修复：确保topics和audience能正确生成）
    text = build_text_for_channel(info)
    # 使用 encode_async 代替直接调用 model.encode()，避免多线程环境下的 PyTorch 线程安全问题
    try:
        vec_array = await encode_async([text], show_progress_bar=False)
        vec = vec_array[0]
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"build_channel_index.py:_process_channel_data:AFTER_EMBEDDING","message":"成功计算向量","data":{"channel_id":channel_id,"vec_shape":list(vec.shape) if hasattr(vec,'shape') else None},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
    except Exception as e:
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"build_channel_index.py:_process_channel_data:EMBEDDING_ERROR","message":"向量编码失败","data":{"channel_id":channel_id,"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        raise

    tags = infer_topics_and_audience(np.expand_dims(vec, axis=0))
    info["topics"] = tags["topics"]
    info["audience"] = tags["audience"]
    
    # 计算互动率和观看率（用于BD评分，避免搜索时重复计算）
    try:
        from core.channel_info import get_recent_videos_stats
        from infrastructure.config import Config
        stats = await get_recent_videos_stats(channel_id, max_results=Config.CHANNEL_INFO["stats_videos_count"], use_for="index")
        subs = info.get("subscriberCount", 0)
        if subs > 0:
            avg_likes = stats.get("avg_likes", 0.0)
            avg_views = stats.get("avg_views", 0.0)
            info["engagement_rate"] = round((avg_likes / subs * 100), 1)
            info["view_rate"] = round((avg_views / subs * 100), 1)
        else:
            info["engagement_rate"] = 0.0
            info["view_rate"] = 0.0
    except Exception as e:
        logger.debug(f"计算频道 {channel_id} 的互动率和观看率失败: {e}")
        info["engagement_rate"] = 0.0
        info["view_rate"] = 0.0
    
    # 检测竞品合作（用于BD评分，避免搜索时重复检测）
    try:
        from core.bd_scoring import detect_competitor_collaborations
        competitor_result = detect_competitor_collaborations(
            info.get("description", ""),
            recent_videos
        )
        info["competitor_detection"] = competitor_result
    except Exception as e:
        logger.debug(f"检测频道 {channel_id} 的竞品合作失败: {e}")
        info["competitor_detection"] = {
            "has_competitor_collab": False,
            "competitors": [],
            "competitor_details": {}
        }

    return (info, vec)


def _validate_channel_data(channel_data: Tuple) -> Tuple | None:
    """
    验证频道数据的有效性
    
    Args:
        channel_data: (channel_id, title, description, subscriber_count, view_count,
                      country, language, emails_json, topics_json, audience_json,
                      recent_videos_json, engagement_rate, view_rate, competitor_detection_json, embedding_bytes)
    
    Returns:
        验证后的数据元组，如果验证失败则返回 None
    """
    # #region agent log
    try:
        import json
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_validate_channel_data:ENTRY","message":"开始验证数据","data":{"has_data":channel_data is not None,"data_length":len(channel_data) if channel_data else 0},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    if not channel_data or len(channel_data) != 15:
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_validate_channel_data:LENGTH_ERROR","message":"数据长度不正确","data":{"expected":15,"actual":len(channel_data) if channel_data else 0},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        return None
    
    ch_id, title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json, \
        recent_videos_json, engagement_rate, view_rate, competitor_detection_json, embedding = channel_data
    
    # 验证必填字段
    if not ch_id or not isinstance(ch_id, str) or not ch_id.strip():
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_validate_channel_data:CHANNEL_ID_ERROR","message":"频道ID验证失败","data":{"ch_id":ch_id,"is_str":isinstance(ch_id, str)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        return None
    
    # 验证数值字段
    try:
        sub_count = int(sub_count) if sub_count is not None else 0
        view_count = int(view_count) if view_count is not None else 0
        # 确保非负
        sub_count = max(0, sub_count)
        view_count = max(0, view_count)
    except (ValueError, TypeError) as e:
        logger.debug(f"验证频道数值字段失败: {e}")
        return None
    
    # 验证字符串字段（清理和截断）
    title = (title or "").strip()[:500] if title else ""  # 限制长度
    desc = (desc or "").strip()[:10000] if desc else ""  # 限制长度
    
    # 验证JSON字段
    try:
        if emails_json:
            emails_list = json.loads(emails_json) if isinstance(emails_json, str) else emails_json
            if not isinstance(emails_list, list):
                emails_json = "[]"
        if topics_json:
            topics_list = json.loads(topics_json) if isinstance(topics_json, str) else topics_json
            if not isinstance(topics_list, list):
                topics_json = "[]"
        if audience_json:
            audience_list = json.loads(audience_json) if isinstance(audience_json, str) else audience_json
            if not isinstance(audience_list, list):
                audience_json = "[]"
        if recent_videos_json:
            recent_videos_list = json.loads(recent_videos_json) if isinstance(recent_videos_json, str) else recent_videos_json
            if not isinstance(recent_videos_list, list):
                recent_videos_json = "[]"
        if competitor_detection_json:
            competitor_detection_dict = json.loads(competitor_detection_json) if isinstance(competitor_detection_json, str) else competitor_detection_json
            if not isinstance(competitor_detection_dict, dict):
                competitor_detection_json = "{}"
    except (json.JSONDecodeError, TypeError) as e:
        # JSON解析失败，使用空数组
        logger.debug(f"解析频道JSON字段失败: {e}，使用空数组")
        emails_json = "[]"
        topics_json = "[]"
        audience_json = "[]"
        recent_videos_json = "[]"
        competitor_detection_json = "{}"
    
    # 验证数值字段（engagement_rate, view_rate）
    try:
        engagement_rate = float(engagement_rate) if engagement_rate is not None else 0.0
        view_rate = float(view_rate) if view_rate is not None else 0.0
        # 确保非负
        engagement_rate = max(0.0, engagement_rate)
        view_rate = max(0.0, view_rate)
    except (ValueError, TypeError) as e:
        logger.debug(f"验证频道互动率和观看率字段失败: {e}")
        engagement_rate = 0.0
        view_rate = 0.0
    
    # 验证向量
    if embedding is None:
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_validate_channel_data:EMBEDDING_NULL","message":"向量为空","data":{"channel_id":ch_id},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        return None
    
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_validate_channel_data:SUCCESS","message":"数据验证成功","data":{"channel_id":ch_id},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    return (ch_id.strip(), title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json,
            recent_videos_json, engagement_rate, view_rate, competitor_detection_json, embedding)


def _batch_upsert_channels(channel_data_list: List[Tuple]) -> None:
    """
    批量插入/更新频道数据到数据库，提高效率。
    使用事务和批量操作优化性能，包含数据验证。
    支持大批量数据自动分批处理，避免单次事务过大。
    
    Args:
        channel_data_list: 列表，每个元素是 (channel_id, title, description, subscriber_count, 
                          view_count, country, language, emails_json, topics_json, audience_json, embedding_bytes)
    """
    if not channel_data_list:
        return
    
    import time
    start_time = time.time()
    
    # 验证所有数据
    validated_data = []
    for data in channel_data_list:
        validated = _validate_channel_data(data)
        if validated:
            validated_data.append(validated)
        else:
            logger.warning(f"频道数据验证失败，跳过: {data[0] if data else 'Unknown'}")
    
    if not validated_data:
        logger.warning("所有频道数据验证失败，没有数据可保存")
        return
    
    # 获取批量大小配置（CP-y2-15：数据库批量操作优化）
    batch_size = Config.DB_BATCH_SIZE
    total_count = len(validated_data)
    
    # 如果数据量小于批量大小，直接处理
    if total_count <= batch_size:
        _execute_batch_upsert(validated_data)
        elapsed = time.time() - start_time
        logger.info(f"批量更新了 {total_count} 个频道到数据库（耗时 {elapsed:.2f} 秒）")
        return
    
    # 大批量数据分批处理
    logger.info(f"开始分批更新 {total_count} 个频道（每批 {batch_size} 个）")
    processed_count = 0
    for i in range(0, total_count, batch_size):
        batch_data = validated_data[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        try:
            _execute_batch_upsert(batch_data)
            processed_count += len(batch_data)
            logger.debug(f"批次 {batch_num}/{total_batches} 完成：已处理 {processed_count}/{total_count} 个频道")
        except Exception as e:
            logger.error(f"批次 {batch_num}/{total_batches} 失败: {e}", exc_info=True)
            # 继续处理下一批，不中断整个流程
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"批量更新完成：共 {processed_count}/{total_count} 个频道（耗时 {elapsed:.2f} 秒，平均 {elapsed/processed_count*1000:.1f} 毫秒/频道）")


def _execute_batch_upsert(validated_data: List[Tuple]) -> None:
    """
    执行单批数据的批量插入/更新操作。
    
    Args:
        validated_data: 已验证的频道数据列表
    """
    from infrastructure.database import get_db_connection
    
    # 使用数据库上下文管理器（CP-y2-15：数据库批量操作优化）
    # 注意：连接池已经提供线程安全，不需要额外的_db_lock（避免死锁）
    with get_db_connection() as conn:
        # 确保schema存在
        _ensure_schema(conn)
        
        # 移除_db_lock，连接池已经提供线程安全
        # 如果确实需要额外保护，应该使用连接级别的锁，而不是全局锁
        cur = conn.cursor()
        
        # 准备批量数据
        channels_data = [
            (ch_id, title, desc, sub_count, view_count, country, lang, emails, topics, audience,
             recent_videos, engagement_rate, view_rate, competitor_detection)
            for ch_id, title, desc, sub_count, view_count, country, lang, emails, topics, audience,
                recent_videos, engagement_rate, view_rate, competitor_detection, _ in validated_data
        ]
        embeddings_data = [
            (ch_id, embedding)
            for ch_id, _, _, _, _, _, _, _, _, _, _, _, _, _, embedding in validated_data
        ]
        
        # 使用事务批量插入频道信息
        cur.executemany(
            """
            INSERT INTO channels (
                channel_id, title, description,
                subscriber_count, view_count,
                country, language,
                emails, topics, audience,
                recent_videos, engagement_rate, view_rate, competitor_detection,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(channel_id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                subscriber_count=excluded.subscriber_count,
                view_count=excluded.view_count,
                country=excluded.country,
                language=excluded.language,
                emails=excluded.emails,
                topics=excluded.topics,
                audience=excluded.audience,
                recent_videos=excluded.recent_videos,
                engagement_rate=excluded.engagement_rate,
                view_rate=excluded.view_rate,
                competitor_detection=excluded.competitor_detection,
                updated_at=CURRENT_TIMESTAMP
            """,
            channels_data
        )
        
        # 批量插入向量
        cur.executemany(
            """
            INSERT INTO channel_embeddings (channel_id, embedding)
            VALUES (?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                embedding=excluded.embedding
            """,
            embeddings_data
        )
        
        # 事务会在上下文管理器退出时自动提交
    
    # 失效相关缓存（在事务外执行，避免阻塞）
    # 注意：validated_data 包含 15 个字段：(channel_id, title, description, subscriber_count, view_count,
    #       country, language, emails_json, topics_json, audience_json, recent_videos_json,
    #       engagement_rate, view_rate, competitor_detection_json, embedding_bytes)
    for ch_id, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in validated_data:
        invalidate_all_channel_caches(ch_id)


def _run_async_in_thread(coro_or_func, *args, max_retries: int = 3, **kwargs):
    """
    在线程中运行异步函数的辅助函数（增强版）。
    为每个线程创建独立的事件循环，避免事件循环冲突。
    包含重试机制和更健壮的错误处理。
    
    Args:
        coro_or_func: 协程对象或异步函数（可调用对象）
        max_retries: 最大重试次数（默认3次）
        *args, **kwargs: 如果 coro_or_func 是可调用对象，这些参数会传递给它
    
    Returns:
        协程的返回值
    
    Raises:
        Exception: 如果所有重试都失败，抛出最后一次的异常
    
    Note:
        - 如果传入协程对象，每次重试时会重新创建协程（通过检查协程状态）
        - 如果传入可调用对象，每次重试时会调用它创建新的协程
    """
    import inspect
    import types
    
    # 判断是协程对象还是可调用对象
    is_coroutine = inspect.iscoroutine(coro_or_func)
    is_coroutine_function = inspect.iscoroutinefunction(coro_or_func)
    
    # 如果是协程对象，需要保存创建它的函数以便重试时重新创建
    # 但我们无法从协程对象获取原始函数，所以需要特殊处理
    if is_coroutine:
        # 协程对象不能重用，我们需要在第一次失败后抛出错误
        # 或者要求调用者传入函数而不是协程对象
        raise ValueError(
            "不能直接传入协程对象，请传入异步函数（可调用对象）。"
            "例如：_run_async_in_thread(_process_channel_data, channel_id, recent_videos_count)"
            "而不是：_run_async_in_thread(_process_channel_data(channel_id, recent_videos_count))"
        )
    
    if not callable(coro_or_func):
        raise TypeError(f"第一个参数必须是协程对象或可调用对象，得到: {type(coro_or_func)}")
    
    loop = None
    last_exception = None
    
    for attempt in range(max_retries):
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_run_async_in_thread:ATTEMPT","message":"尝试运行异步函数","data":{"attempt":attempt+1,"max_retries":max_retries,"thread":threading.current_thread().name,"func_name":getattr(coro_or_func,'__name__','unknown')},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        try:
            # 确保之前的事件循环已清理
            try:
                old_loop = asyncio.get_event_loop()
                if old_loop and not old_loop.is_closed():
                    # #region agent log
                    try:
                        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_run_async_in_thread:CLOSE_OLD_LOOP","message":"关闭旧事件循环","data":{"thread":threading.current_thread().name},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
                    try:
                        old_loop.close()
                    except Exception:
                        pass
            except RuntimeError:
                # 没有运行的事件循环，这是正常的
                pass
            
            # 为当前线程创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_run_async_in_thread:NEW_LOOP","message":"创建新事件循环","data":{"thread":threading.current_thread().name,"loop_id":id(loop)},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            
            # 每次重试时创建新的协程对象
            coro = coro_or_func(*args, **kwargs)
            if not inspect.iscoroutine(coro):
                raise TypeError(f"函数 {coro_or_func} 必须返回协程对象")
            
            # 运行协程
            result = loop.run_until_complete(coro)
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_run_async_in_thread:SUCCESS","message":"异步函数执行成功","data":{"thread":threading.current_thread().name,"attempt":attempt+1},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            return result
            
        except YouTubeQuotaExceededError:
            # 配额错误应该立即传播，不重试
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_run_async_in_thread:QUOTA_ERROR","message":"配额已用完，立即传播","data":{"thread":threading.current_thread().name,"attempt":attempt+1},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            raise  # 立即重新抛出，不重试
        except (RuntimeError, ValueError) as e:
            # 事件循环相关错误，可能是清理不彻底导致的
            last_exception = e
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_run_async_in_thread:LOOP_ERROR","message":"事件循环错误","data":{"thread":threading.current_thread().name,"attempt":attempt+1,"error":str(e),"error_type":type(e).__name__,"will_retry":attempt < max_retries - 1},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            if attempt < max_retries - 1:
                logger.debug(f"事件循环错误（尝试 {attempt + 1}/{max_retries}），重试中: {e}")
                time.sleep(0.1 * (attempt + 1))  # 指数退避
                continue
            else:
                logger.error(f"事件循环错误，已重试 {max_retries} 次: {e}")
                raise
        except Exception as e:
            # 其他错误，不重试（可能是业务逻辑错误）
            last_exception = e
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_run_async_in_thread:OTHER_ERROR","message":"其他异常","data":{"thread":threading.current_thread().name,"attempt":attempt+1,"error":str(e),"error_type":type(e).__name__,"is_network_error":isinstance(e, (ConnectionError, TimeoutError)),"will_retry":attempt < max_retries - 1 and isinstance(e, (ConnectionError, TimeoutError))},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            if attempt < max_retries - 1 and isinstance(e, (ConnectionError, TimeoutError)):
                # 网络错误可以重试，使用指数退避（1s, 2s, 4s...）
                wait_time = min(2 ** attempt, 10)  # 最多等待10秒
                logger.warning(f"网络错误（尝试 {attempt + 1}/{max_retries}），{wait_time} 秒后重试: {e}")
                time.sleep(wait_time)
                continue
            else:
                # 网络错误重试失败后，记录但不抛出，让上层决定是否继续
                logger.error(f"网络错误，已重试 {max_retries} 次仍失败: {e}")
                raise
        finally:
            # 清理事件循环（每次尝试后都清理）
            if loop is not None:
                try:
                    # 取消所有待处理的任务
                    if not loop.is_closed():
                        try:
                            # Python 3.7+
                            pending = asyncio.all_tasks(loop)
                        except AttributeError:
                            # Python 3.6 兼容
                            pending = asyncio.Task.all_tasks(loop)
                        except RuntimeError:
                            pending = []
                        
                        if pending:
                            for task in pending:
                                if not task.done():
                                    task.cancel()
                            # 等待所有任务完成或取消（带超时）
                            try:
                                not_done = [t for t in pending if not t.done()]
                                if not_done:
                                    # 使用超时避免无限等待
                                    loop.run_until_complete(
                                        asyncio.wait_for(
                                            asyncio.gather(*not_done, return_exceptions=True),
                                            timeout=5.0
                                        )
                                    )
                            except (RuntimeError, ValueError, asyncio.TimeoutError):
                                pass
                except Exception as cleanup_err:
                    logger.debug(f"清理待处理任务时出错: {cleanup_err}")
                
                # 关闭事件循环
                try:
                    if not loop.is_closed():
                        loop.close()
                except Exception as close_err:
                    logger.debug(f"关闭事件循环时出错: {close_err}")
                
                # 移除当前线程的事件循环引用
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
    
    # 如果所有重试都失败
    if last_exception:
        raise last_exception
    raise RuntimeError("未知错误：所有重试都失败")


def _process_channel_worker(channel_id: str, skip_if_recent: bool, recent_videos_count: int, max_age_days: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    工作线程函数：处理单个频道（增强版）。
    包含重试机制和更健壮的错误处理。
    
    Returns:
        (success: bool, data: tuple | None) - 如果成功，返回 (True, channel_data)，否则返回 (False, None)
    """
    # 使用连接池而不是直接连接（统一连接管理，避免阻塞）
    from infrastructure.database import get_db_connection
    
    max_retries = 2  # 最多重试2次
    last_exception = None
    
    for attempt in range(max_retries):
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:_process_channel_worker:ATTEMPT","message":"工作线程尝试处理频道","data":{"channel_id":channel_id,"attempt":attempt+1,"max_retries":max_retries,"thread":threading.current_thread().name},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        try:
            # 使用连接池检查是否需要更新（修复作用域问题）
            try:
                with get_db_connection() as conn:
                    _ensure_schema(conn)
                    # 检查是否需要更新（在with块内完成）
                    if skip_if_recent and not _channel_needs_update(conn, channel_id, max_age_days):
                        # #region agent log
                        try:
                            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:_process_channel_worker:SKIP","message":"跳过更新，数据还新鲜","data":{"channel_id":channel_id},"timestamp":int(time.time()*1000)}) + '\n')
                        except: pass
                        # #endregion
                        return (False, None)  # 跳过，数据还新鲜
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as db_err:
                # 数据库错误，如果是最后一次尝试则抛出
                last_exception = db_err
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:_process_channel_worker:DB_ERROR","message":"数据库错误","data":{"channel_id":channel_id,"attempt":attempt+1,"error":str(db_err),"error_type":type(db_err).__name__,"will_retry":attempt < max_retries - 1},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                if attempt < max_retries - 1:
                    logger.debug(f"数据库错误（尝试 {attempt + 1}/{max_retries}），重试中: {db_err}")
                    time.sleep(0.2 * (attempt + 1))
                    continue
                else:
                    logger.warning(f"处理频道 {channel_id} 时数据库错误: {db_err}")
                    return (False, None)
            
            # 处理频道数据（获取信息、计算向量和标签）
            # 在工作线程中运行异步函数（使用独立事件循环）
            # 注意：传入函数对象而不是协程对象，以便重试时能重新创建协程
            # 增加重试次数以应对网络不稳定（DNS解析失败等）
            try:
                info, vec = _run_async_in_thread(_process_channel_data, max_retries=3, channel_id=channel_id, recent_videos_count=recent_videos_count)
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_process_channel_worker:ASYNC_SUCCESS","message":"异步处理成功","data":{"channel_id":channel_id,"has_info":info is not None,"has_vec":vec is not None},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
            except YouTubeQuotaExceededError:
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_worker:QUOTA_ERROR","message":"配额已用完","data":{"channel_id":channel_id},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                raise  # 重新抛出，让上层知道需要停止
            except (RuntimeError, ValueError) as loop_err:
                # 事件循环错误，重试
                last_exception = loop_err
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:_process_channel_worker:LOOP_ERROR","message":"事件循环错误","data":{"channel_id":channel_id,"attempt":attempt+1,"error":str(loop_err),"error_type":type(loop_err).__name__,"will_retry":attempt < max_retries - 1},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                if attempt < max_retries - 1:
                    logger.debug(f"事件循环错误（尝试 {attempt + 1}/{max_retries}），重试中: {loop_err}")
                    time.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    logger.warning(f"处理频道 {channel_id} 时事件循环错误: {loop_err}")
                    return (False, None)
            
            if info is None or vec is None:
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_worker:NULL_RESULT","message":"处理结果为空","data":{"channel_id":channel_id,"info_is_none":info is None,"vec_is_none":vec is None},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                return (False, None)
            
            # 准备数据用于批量插入
            try:
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_process_channel_worker:BEFORE_PREPARE","message":"准备数据前检查","data":{"channel_id":channel_id,"has_title":"title" in info,"has_description":"description" in info,"subscriber_count":info.get("subscriberCount"),"view_count":info.get("viewCount")},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                channel_data = (
                    channel_id,
                    info.get("title", ""),
                    info.get("description", ""),
                    int(info.get("subscriberCount") or 0),
                    int(info.get("viewCount") or 0),
                    info.get("country"),
                    (info.get("defaultLanguage") or info.get("defaultAudioLanguage")),
                    json.dumps(info.get("emails", []), ensure_ascii=False),
                    json.dumps(info.get("topics", []), ensure_ascii=False),
                    json.dumps(info.get("audience", []), ensure_ascii=False),
                    json.dumps(info.get("recent_videos", []), ensure_ascii=False),  # 保存最近视频
                    float(info.get("engagement_rate", 0.0)),  # 保存互动率
                    float(info.get("view_rate", 0.0)),  # 保存观看率
                    json.dumps(info.get("competitor_detection", {}), ensure_ascii=False),  # 保存竞品检测结果
                    vec.astype(np.float32).tobytes(),
                )
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_process_channel_worker:AFTER_PREPARE","message":"数据准备成功","data":{"channel_id":channel_id,"data_length":len(channel_data)},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
            except (ValueError, TypeError, KeyError) as data_err:
                # #region agent log
                try:
                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:_process_channel_worker:DATA_PREPARE_ERROR","message":"数据准备失败","data":{"channel_id":channel_id,"error":str(data_err),"error_type":type(data_err).__name__},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                logger.warning(f"准备频道 {channel_id} 数据时出错: {data_err}")
                return (False, None)
            
            return (True, channel_data)
            
        except YouTubeQuotaExceededError:
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_worker:QUOTA_EXCEEDED","message":"配额已用完","data":{"channel_id":channel_id},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            raise  # 重新抛出，让上层知道需要停止
        except Exception as e:
            last_exception = e
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:_process_channel_worker:UNHANDLED_ERROR","message":"未处理的异常","data":{"channel_id":channel_id,"attempt":attempt+1,"error":str(e),"error_type":type(e).__name__,"is_network_error":isinstance(e, (ConnectionError, TimeoutError, OSError)),"will_retry":attempt < max_retries - 1 and isinstance(e, (ConnectionError, TimeoutError, OSError))},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            if attempt < max_retries - 1:
                # 网络错误或临时错误可以重试，使用指数退避
                if isinstance(e, (ConnectionError, TimeoutError, OSError)):
                    wait_time = min(2 ** attempt, 10)  # 最多等待10秒
                    logger.warning(f"处理频道 {channel_id} 时网络错误（尝试 {attempt + 1}/{max_retries}），{wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    # 其他错误不重试
                    logger.warning(f"处理频道 {channel_id} 失败: {e}")
                    return (False, None)
            else:
                # 网络错误重试失败后，跳过该频道继续处理其他频道
                if isinstance(e, (ConnectionError, TimeoutError, OSError)):
                    logger.error(f"处理频道 {channel_id} 时网络错误，已重试 {max_retries} 次仍失败，跳过该频道: {e}")
                else:
                    logger.warning(f"处理频道 {channel_id} 失败（已重试 {max_retries} 次）: {e}")
                return (False, None)
    
    # 如果所有重试都失败
    if last_exception:
        logger.warning(f"处理频道 {channel_id} 失败（所有重试都失败）: {last_exception}")
    return (False, None)


def build_index(
    seed_channel_ids: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    max_age_days: int = 60,
    recent_videos_count: int = 2,
    max_workers: int | None = None,
    batch_size: int = 20,
    max_channels_to_update: int | None = 200,
    force_refresh_keywords: bool = False,
) -> None:
    """
    构建/更新本地加密货币频道索引（优化版：支持并行处理和批量更新）。
    
    Args:
        seed_channel_ids: 你手动收集的一批优质币圈频道 ID。
        keywords: 用于搜索频道的关键词。
        max_age_days: 频道数据超过多少天需要更新（默认 60 天，减少API消耗）。
        recent_videos_count: 获取每个频道最近多少个视频（默认 3，原来 5，可减少配额消耗）。
        max_workers: 并行处理的线程数（如果为None，则使用Config中的配置）。
        batch_size: 批量提交到数据库的频道数量（默认 20）。
        force_refresh_keywords: 是否强制刷新关键词搜索（忽略缓存），默认False。
    """
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            keywords_list = list(keywords) if keywords else []
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ENTRY","location":"build_channel_index.py:build_index:ENTRY","message":"build_index函数入口","data":{"has_seed":seed_channel_ids is not None,"has_keywords":keywords is not None,"keywords_count":len(keywords_list) if keywords else 0,"force_refresh":force_refresh_keywords},"timestamp":int(time.time()*1000)}) + '\n')
    except Exception as e:
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ENTRY","location":"build_channel_index.py:build_index:ENTRY_ERROR","message":"build_index入口日志错误","data":{"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
    # #endregion
    # 使用配置化的线程池大小
    if max_workers is None:
        max_workers = Config.get_thread_pool_size("index_build_workers", Config.CONCURRENT_PROCESSING["index_build_workers"])
    
    logger.info("=" * 60)
    logger.info("开始构建/更新频道索引")
    logger.info(f"配置: 线程数={max_workers}, 批量大小={batch_size}, 最大更新数={max_channels_to_update}")
    logger.info("=" * 60)
    
    # 添加调试日志到文件
    log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
    try:
        import json
        import time
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": int(time.time() * 1000),
                "location": "build_channel_index.py:build_index",
                "message": "开始构建/更新频道索引",
                "data": {"max_workers": max_workers, "batch_size": batch_size, "max_channels_to_update": max_channels_to_update},
                "sessionId": "debug-session",
                "runId": "build-index",
                "hypothesisId": "START"
            }, ensure_ascii=False) + "\n")
    except: pass
    
    # 使用连接池统一管理连接
    from infrastructure.database import get_db_connection
    
    # 初始化schema（使用临时连接，添加日志）
    logger.info("正在初始化数据库连接和表结构...")
    try:
        with get_db_connection() as init_conn:
            _ensure_schema(init_conn)
        logger.info("数据库初始化完成")
        # 添加调试日志
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "build_channel_index.py:build_index",
                    "message": "数据库初始化完成",
                    "data": {},
                    "sessionId": "debug-session",
                    "runId": "build-index",
                    "hypothesisId": "START"
                }, ensure_ascii=False) + "\n")
        except: pass
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}", exc_info=True)
        # 添加错误日志
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "build_channel_index.py:build_index",
                    "message": "数据库初始化失败",
                    "data": {"error": str(e)},
                    "sessionId": "debug-session",
                    "runId": "build-index",
                    "hypothesisId": "START"
                }, ensure_ascii=False) + "\n")
        except: pass
        raise
    
    seen: set[str] = set()

    # 尝试从缓存加载已发现的频道ID
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            keywords_list = list(keywords) if keywords else []
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:build_index:CACHE_CHECK","message":"检查缓存状态","data":{"has_keywords":keywords is not None,"keywords_count":len(keywords_list),"force_refresh":force_refresh_keywords},"timestamp":int(time.time()*1000)}) + '\n')
    except Exception as e:
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:build_index:CACHE_CHECK_ERROR","message":"缓存检查日志错误","data":{"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
    # #endregion
    cached_ids, cached_keywords_hash, cached_last_updated = _load_cached_channel_ids()
    current_keywords_hash = _get_keywords_hash(keywords)
    use_cache = not force_refresh_keywords and not _should_refresh_cache(cached_keywords_hash, current_keywords_hash)
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"build_channel_index.py:build_index:CACHE_DECISION","message":"缓存决策","data":{"use_cache":use_cache,"cached_hash":cached_keywords_hash,"current_hash":current_keywords_hash,"cached_ids_count":len(cached_ids) if cached_ids else 0},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    
    all_ids: List[str] = []
    if seed_channel_ids:
        seed_list = list(seed_channel_ids)
        all_ids.extend(seed_list)
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:SEED_ADDED","message":"添加种子频道","data":{"seed_count":len(seed_list),"all_ids_count":len(all_ids)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
    
    if use_cache and cached_ids:
        logger.info(f"使用缓存的频道ID列表（共 {len(cached_ids)} 个），跳过关键词搜索以节省配额")
        if cached_last_updated:
            logger.info(f"缓存更新时间: {cached_last_updated}")
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:BEFORE_CACHE_MERGE","message":"合并缓存前","data":{"all_ids_count":len(all_ids),"cached_ids_count":len(cached_ids),"all_ids_set_size":len(set(all_ids)),"cached_not_in_all_ids":len([ch_id for ch_id in cached_ids if ch_id not in all_ids])},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        # 合并缓存ID时去重，避免与seed_channel_ids重复
        new_cached_ids = [ch_id for ch_id in cached_ids if ch_id not in all_ids]
        all_ids.extend(new_cached_ids)
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:AFTER_CACHE_MERGE","message":"合并缓存后","data":{"all_ids_count":len(all_ids),"all_ids_set_size":len(set(all_ids)),"has_duplicates":len(all_ids) != len(set(all_ids))},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
    elif keywords:
        # 需要重新搜索关键词
        logger.info(f"正在通过关键词搜索频道: {', '.join(keywords)}")
        discovered_ids = []
        for kw in keywords:
            # 使用辅助函数统一处理，避免事件循环状态检查的复杂性
            # 在 Python 3.10+ 中，直接使用 _run_async_in_thread 更安全
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX","location":"build_channel_index.py:761","message":"Using _run_async_in_thread for keyword search","data":{"thread":threading.current_thread().name,"keyword":kw},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            ids = _run_async_in_thread(_discover_channels_by_keyword, kw, max_results=30, max_retries=3)
            if ids:
                logger.info(f"关键词 '{kw}': 找到 {len(ids)} 个频道")
            discovered_ids.extend(ids)
        
        # 合并缓存中的ID和新发现的ID（去重）
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:BEFORE_DISCOVERED_MERGE","message":"合并新发现ID前","data":{"all_ids_count":len(all_ids),"discovered_ids_count":len(discovered_ids),"cached_ids_count":len(cached_ids) if cached_ids else 0},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        all_ids.extend(discovered_ids)
        if cached_ids:
            # 合并时去重：检查不在discovered_ids中，也不在all_ids中（避免与seed_channel_ids重复）
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:BEFORE_CACHED_MERGE","message":"合并缓存ID前","data":{"all_ids_count":len(all_ids),"all_ids_set_size":len(set(all_ids)),"cached_ids_count":len(cached_ids),"cached_not_in_discovered":len([ch_id for ch_id in cached_ids if ch_id not in discovered_ids]),"cached_not_in_all_ids":len([ch_id for ch_id in cached_ids if ch_id not in all_ids])},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            new_ids = [ch_id for ch_id in cached_ids if ch_id not in discovered_ids and ch_id not in all_ids]
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:AFTER_CACHED_MERGE","message":"合并缓存ID后","data":{"new_ids_count":len(new_ids),"new_ids_not_in_all_ids":len([ch_id for ch_id in new_ids if ch_id not in all_ids]),"all_ids_count":len(all_ids),"all_ids_set_size":len(set(all_ids))},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            all_ids.extend(new_ids)
            logger.info(f"合并缓存和新发现的频道ID（缓存: {len(cached_ids)} 个，新发现: {len(discovered_ids)} 个，去重后总计: {len(all_ids)} 个）")
        else:
            logger.info(f"新发现 {len(discovered_ids)} 个频道ID")
        
        # 保存到缓存（只保存关键词搜索发现的ID，不包括seed_channel_ids）
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:build_index:BEFORE_SAVE_CACHE","message":"保存缓存前","data":{"all_ids_count":len(all_ids),"discovered_ids_count":len(discovered_ids),"cached_ids_count":len(cached_ids) if cached_ids else 0,"will_save_seeds":bool(seed_channel_ids)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        # 只保存关键词搜索发现的ID和缓存中的ID，不包括seed_channel_ids
        ids_to_cache = list(set(discovered_ids + (cached_ids if cached_ids else [])))
        _save_cached_channel_ids(ids_to_cache, current_keywords_hash)
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:build_index:AFTER_SAVE_CACHE","message":"保存缓存后","data":{"saved_ids_count":len(ids_to_cache)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion

    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:build_index:BEFORE_UNIQUE","message":"去重前","data":{"all_ids_count":len(all_ids),"all_ids_set_size":len(set(all_ids)),"seen_size":len(seen)},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    # 使用set去重（seen集合未使用，直接用set更高效）
    unique_ids = list(set(all_ids))
    # #region agent log
    try:
        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"build_channel_index.py:build_index:AFTER_UNIQUE","message":"去重后","data":{"unique_ids_count":len(unique_ids),"unique_ids_set_size":len(set(unique_ids)),"has_duplicates":len(unique_ids) != len(set(unique_ids))},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    
    # 检查有多少频道需要更新（批量检查以提高效率）
    logger.info(f"正在检查 {len(unique_ids)} 个频道是否需要更新...")
    needs_update = []
    skipped = 0
    
    # 使用连接池检查频道更新状态
    with get_db_connection() as check_conn:
        for ch_id in unique_ids:
            if _channel_needs_update(check_conn, ch_id, max_age_days=max_age_days):
                needs_update.append(ch_id)
            else:
                skipped += 1
    
    logger.info(f"频道统计: 总计 {len(unique_ids)} 个，需要更新 {len(needs_update)} 个，跳过 {skipped} 个")
    if max_channels_to_update is not None and len(needs_update) > max_channels_to_update:
        needs_update = needs_update[:max_channels_to_update]
        logger.info(f"为控制配额，本次仅处理前 {max_channels_to_update} 个需要更新的频道")
    # 估算配额：channels(1) + playlistItems(约100) ~ 101，取整到 100
    logger.info(f"配额使用估算: 约 {len(needs_update) * 100} 单位（每个频道 ~100 单位）")
    logger.info(f"并行处理配置: 线程数 {max_workers}，批量大小 {batch_size}")
    
    if not needs_update:
        logger.info("所有频道数据都是最新的，无需更新！")
        return
    
    logger.info(f"开始并行处理 {len(needs_update)} 个需要更新的频道...")
    
    processed_count = 0
    failed_count = 0
    quota_exceeded = False
    quota_used = 0
    start_time = time.time()
    
    # 批量收集处理结果
    batch_data: List[tuple] = []
    
    # 使用线程池并行处理
    executor = None
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_channel = {
                executor.submit(_process_channel_worker, ch_id, False, recent_videos_count, max_age_days): ch_id
                for ch_id in needs_update
            }
            
            # 处理完成的任务
            for idx, future in enumerate(as_completed(future_to_channel), 1):
                ch_id = future_to_channel[future]
                try:
                    # 添加超时保护（避免单个任务卡死）
                    try:
                        success, channel_data = future.result(timeout=300)  # 5分钟超时
                    except TimeoutError:
                        logger.warning(f"处理频道 {ch_id} 超时（5分钟），跳过")
                        failed_count += 1
                        continue
                    
                    if success and channel_data:
                        processed_count += 1
                        quota_used += 1 + 100  # 估算配额使用
                        batch_data.append(channel_data)
                        
                        # 批量提交到数据库
                        if len(batch_data) >= batch_size:
                            # #region agent log
                            try:
                                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:BEFORE_BATCH_UPSERT","message":"准备批量更新数据库","data":{"batch_size":len(batch_data)},"timestamp":int(time.time()*1000)}) + '\n')
                            except: pass
                            # #endregion
                            try:
                                _batch_upsert_channels(batch_data)
                                batch_data.clear()
                                # #region agent log
                                try:
                                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:AFTER_BATCH_UPSERT","message":"批量更新成功","data":{},"timestamp":int(time.time()*1000)}) + '\n')
                                except: pass
                                # #endregion
                            except Exception as db_err:
                                # #region agent log
                                try:
                                    with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"build_channel_index.py:build_index:BATCH_UPSERT_ERROR","message":"批量更新失败","data":{"error":str(db_err),"error_type":type(db_err).__name__},"timestamp":int(time.time()*1000)}) + '\n')
                                except: pass
                                # #endregion
                                logger.error(f"批量更新数据库失败: {db_err}，保留数据待后续提交")
                                # 不清空 batch_data，等待后续提交
                    
                    else:
                        failed_count += 1
                    
                    # 进度报告
                    if idx % 10 == 0 or idx == len(needs_update):
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (len(needs_update) - idx) / rate if rate > 0 else 0
                        logger.info(
                            f"进度: {idx}/{len(needs_update)} | "
                            f"成功: {processed_count} | 失败: {failed_count} | "
                            f"速度: {rate:.1f} 频道/秒 | "
                            f"预计剩余: {remaining:.0f} 秒 | "
                            f"已用配额: ~{quota_used}"
                        )
                    
                except YouTubeQuotaExceededError:
                    quota_exceeded = True
                    # #region agent log
                    try:
                        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:build_index:QUOTA_EXCEEDED","message":"配额已用完，停止处理","data":{"processed_count":processed_count,"quota_used":quota_used},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
                    logger.error(
                        f"YouTube API 配额已用完，已成功处理 {processed_count} 个频道。"
                        f"已使用约 {quota_used} 单位配额。"
                    )
                    # 取消未完成的任务
                    for f in future_to_channel:
                        if not f.done():
                            f.cancel()
                    break
                except Exception as e:
                    failed_count += 1
                    # #region agent log
                    try:
                        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"build_channel_index.py:build_index:FUTURE_ERROR","message":"处理频道时出错","data":{"channel_id":ch_id,"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
                    logger.error(f"处理频道 {ch_id} 时出错: {e}", exc_info=True)
                    continue
    except KeyboardInterrupt:
        logger.warning("用户中断，正在清理资源...")
        quota_exceeded = True
        # 尝试提交已处理的数据
        if batch_data:
            try:
                _batch_upsert_channels(batch_data)
            except Exception as e:
                logger.error(f"中断时提交数据失败: {e}")
    except Exception as e:
        logger.error(f"线程池执行出错: {e}", exc_info=True)
        # 尝试提交已处理的数据
        if batch_data:
            try:
                _batch_upsert_channels(batch_data)
            except Exception as db_err:
                logger.error(f"错误时提交数据失败: {db_err}")
    finally:
        # 确保所有资源都被清理
        if executor:
            try:
                # 等待所有任务完成或取消（带超时）
                executor.shutdown(wait=True, timeout=30)
            except Exception as e:
                logger.debug(f"关闭线程池时出错: {e}")
    
    # 提交剩余的批量数据
    if batch_data:
        _batch_upsert_channels(batch_data)
    
    elapsed_time = time.time() - start_time
    if not quota_exceeded:
        logger.info(
            f"完成！已处理: {processed_count} 个频道，失败: {failed_count} 个，"
            f"已用配额: 约 {quota_used} 单位，总耗时: {elapsed_time:.1f} 秒，"
            f"平均速度: {processed_count / elapsed_time:.2f} 频道/秒" if elapsed_time > 0 else ""
        )


if __name__ == "__main__":
    # 全局异常处理和资源清理
    exit_code = 0
    try:
        print("=" * 60, flush=True)
        print("YouTube频道索引构建脚本", flush=True)
        print("=" * 60, flush=True)
        
        # 添加启动日志
        log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
        try:
            import json
            import time
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "build_channel_index.py:__main__",
                    "message": "脚本开始执行",
                    "data": {},
                    "sessionId": "debug-session",
                    "runId": "build-index",
                    "hypothesisId": "START"
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"警告：无法写入启动日志: {e}", flush=True)
        
        print("正在初始化...", flush=True)
        # 你可以在这里填写你自己常看的/认为优质的币圈频道链接或 ID 作为种子。
        raw_seed_channels: List[str] = [
            "https://www.youtube.com/@Crypto621",
            "https://www.youtube.com/@bitraderx",
            "https://www.youtube.com/@speculation",
        ]

        seed_channels: List[str] = []
        for s in raw_seed_channels:
            try:
                if s.startswith("http"):
                    # 使用辅助函数统一处理，避免事件循环状态检查的复杂性
                    # #region agent log
                    try:
                        with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX","location":"build_channel_index.py:914","message":"Using _run_async_in_thread for seed channel","data":{"thread":threading.current_thread().name,"url":s},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
                    cid = _run_async_in_thread(extract_channel_id_from_url, s, use_for="index", max_retries=3)
                else:
                    cid = s
                seed_channels.append(cid)
            except Exception as e:
                logger.warning(f"无法解析种子频道 {s}: {e}")

        # 关键词列表：与搜索库（business_rules.py）对齐，重点关注BD模式核心需求
        # 包含：核心交易类型、分析方法、主流币种、多语言支持、重要受众关键词
        default_keywords = [
            # 主流币种（基础）
            "bitcoin",
            "ethereum",
            "altcoins",
            
            # 核心交易类型（BD模式重点关注）
            "crypto trading",
            "futures trading",
            "leverage trading",
            "perpetual contracts",
            "margin trading",
            "crypto scalping",
            "spot trading",
            "swing trading",
            "day trading",
            "copy trading",
            
            # 分析方法（BD模式强相关）
            "technical analysis",
            "crypto trading signals",
            "whale watching",
            "liquidation analysis",
            "market analysis",
            
            # 其他重要主题
            "crypto news",
            "defi",
            "airdrop hunting",
            
            # 多语言支持（西班牙语市场）
            "criptomonedas",
            "mercado cripto",
            "trading crypto",  # 英语变体
            
            # 受众关键词（帮助发现目标频道）
            "active futures traders",
            "leverage traders",
            "day traders",
        ]

        build_index(
            seed_channel_ids=seed_channels, 
            keywords=default_keywords,
            max_workers=5,  # 可以根据你的网络和 API 配额调整
            batch_size=20,  # 批量提交大小
        )
        
        print("=" * 60, flush=True)
        print("脚本执行完成", flush=True)
        print("=" * 60, flush=True)
        
    except KeyboardInterrupt:
        print("\n用户中断，正在清理资源...", flush=True)
        logger.warning("脚本被用户中断")
        exit_code = 130  # 标准中断退出码
    except YouTubeQuotaExceededError as e:
        print(f"\n错误：YouTube API 配额已用完: {e}", flush=True)
        logger.error(f"YouTube API 配额已用完: {e}")
        exit_code = 1
    except Exception as e:
        print(f"\n错误：脚本执行失败: {e}", flush=True)
        logger.error(f"脚本执行失败: {e}", exc_info=True)
        exit_code = 1
    finally:
        # 清理资源
        try:
            # 清理所有事件循环引用
            try:
                loop = asyncio.get_event_loop()
                if loop and not loop.is_closed():
                    try:
                        loop.close()
                    except Exception:
                        pass
            except RuntimeError:
                pass
            asyncio.set_event_loop(None)
        except Exception as cleanup_err:
            logger.debug(f"清理资源时出错: {cleanup_err}")
        
        # 清理数据库连接池（如果需要）
        try:
            from infrastructure.database import _connection_pool, _pool_initialized
            if _pool_initialized:
                # 关闭连接池中的所有连接
                while not _connection_pool.empty():
                    try:
                        conn = _connection_pool.get_nowait()
                        conn.close()
                    except Exception:
                        pass
        except Exception as pool_err:
            logger.debug(f"清理连接池时出错: {pool_err}")
        
        # 退出
        sys.exit(exit_code)


