import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np

from candidate_collection import search_candidate_channels_by_title
from cache import invalidate_all_channel_caches
from channel_info import (
    get_channel_basic_info,
    get_recent_video_snippets_for_channel,
)
from channel_parser import extract_channel_id_from_url
from config import Config
from embedding import (
    get_embed_model,
    infer_topics_and_audience,
)
from logger import get_logger
from utils import build_text_for_channel, extract_emails_from_text
from youtube_api import YouTubeQuotaExceededError, yt_get

logger = get_logger()


DB_PATH = os.path.join(os.path.dirname(__file__), "channel_index.db")

# 线程安全的锁，用于数据库操作
_db_lock = Lock()


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


def _discover_channels_by_keyword(keyword: str, max_results: int = 30) -> List[str]:
    """
    通过关键词搜索发现频道 ID（只搜索一次）。
    默认 max_results=30 以减少配额消耗（原来 50）。
    """
    try:
        ids = search_candidate_channels_by_title(keyword, limit=max_results, use_for="index")
        return ids
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"关键词 '{keyword}' 搜索失败（网络错误）: {e}")
        return []
    except Exception as e:
        logger.warning(f"关键词 '{keyword}' 搜索失败: {e}")
        return []


def _channel_needs_update(conn: sqlite3.Connection, channel_id: str, max_age_days: int = 7) -> bool:
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


def _process_channel_data(channel_id: str, recent_videos_count: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
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
    try:
        info = get_channel_basic_info(channel_id, use_for="index")
    except YouTubeQuotaExceededError as e:
        logger.error(f"YouTube API 配额已用完: {e}")
        raise
    except Exception as e:
        logger.warning(f"获取频道 {channel_id} 失败: {e}")
        return (None, None)

    # 获取最近视频（用于更精准的 embedding）
    try:
        recent_videos = get_recent_video_snippets_for_channel(channel_id, max_results=recent_videos_count, use_for="index")
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
    model = get_embed_model()
    text = build_text_for_channel(info)
    vec = model.encode([text], convert_to_numpy=True)[0]

    tags = infer_topics_and_audience(np.expand_dims(vec, axis=0))
    info["topics"] = tags["topics"]
    info["audience"] = tags["audience"]

    return (info, vec)


def _validate_channel_data(channel_data: Tuple) -> Tuple | None:
    """
    验证频道数据的有效性
    
    Args:
        channel_data: (channel_id, title, description, subscriber_count, view_count,
                      country, language, emails_json, topics_json, audience_json, embedding_bytes)
    
    Returns:
        验证后的数据元组，如果验证失败则返回 None
    """
    if not channel_data or len(channel_data) != 11:
        return None
    
    ch_id, title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json, embedding = channel_data
    
    # 验证必填字段
    if not ch_id or not isinstance(ch_id, str) or not ch_id.strip():
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
    except (json.JSONDecodeError, TypeError) as e:
        # JSON解析失败，使用空数组
        logger.debug(f"解析频道JSON字段失败: {e}，使用空数组")
        emails_json = "[]"
        topics_json = "[]"
        audience_json = "[]"
    
    # 验证向量
    if embedding is None:
        return None
    
    return (ch_id.strip(), title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json, embedding)


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
    from database import get_db_connection
    
    # 使用数据库上下文管理器（CP-y2-15：数据库批量操作优化）
    with get_db_connection() as conn:
        # 确保schema存在
        _ensure_schema(conn)
        
        with _db_lock:
            cur = conn.cursor()
            
            # 准备批量数据
            channels_data = [
                (ch_id, title, desc, sub_count, view_count, country, lang, emails, topics, audience)
                for ch_id, title, desc, sub_count, view_count, country, lang, emails, topics, audience, _ in validated_data
            ]
            embeddings_data = [
                (ch_id, embedding)
                for ch_id, _, _, _, _, _, _, _, _, _, embedding in validated_data
            ]
            
            # 使用事务批量插入频道信息
            cur.executemany(
                """
                INSERT INTO channels (
                    channel_id, title, description,
                    subscriber_count, view_count,
                    country, language,
                    emails, topics, audience, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
    for ch_id, _, _, _, _, _, _, _, _, _, _ in validated_data:
        invalidate_all_channel_caches(ch_id)


def _process_channel_worker(channel_id: str, skip_if_recent: bool, recent_videos_count: int, max_age_days: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    工作线程函数：处理单个频道。
    
    Returns:
        (success: bool, data: tuple | None) - 如果成功，返回 (True, channel_data)，否则返回 (False, None)
    """
    # 每个线程使用独立的数据库连接（仅用于检查是否需要更新）
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    
    try:
        # 检查是否需要更新
        if skip_if_recent and not _channel_needs_update(conn, channel_id, max_age_days):
            conn.close()
            return (False, None)  # 跳过，数据还新鲜
        
        conn.close()
        
        # 处理频道数据（获取信息、计算向量和标签）
        info, vec = _process_channel_data(channel_id, recent_videos_count)
        
        if info is None or vec is None:
            return (False, None)
        
        # 准备数据用于批量插入
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
            vec.astype(np.float32).tobytes(),
        )
        
        return (True, channel_data)
        
    except YouTubeQuotaExceededError:
        conn.close()
        raise  # 重新抛出，让上层知道需要停止
    except Exception as e:
        conn.close()
        logger.warning(f"处理频道 {channel_id} 失败: {e}")
        return (False, None)


def build_index(
    seed_channel_ids: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    max_age_days: int = 7,
    recent_videos_count: int = 3,
    max_workers: int | None = None,
    batch_size: int = 20,
) -> None:
    """
    构建/更新本地加密货币频道索引（优化版：支持并行处理和批量更新）。
    
    Args:
        seed_channel_ids: 你手动收集的一批优质币圈频道 ID。
        keywords: 用于搜索频道的关键词。
        max_age_days: 频道数据超过多少天需要更新（默认 7 天）。
        recent_videos_count: 获取每个频道最近多少个视频（默认 3，原来 5，可减少配额消耗）。
        max_workers: 并行处理的线程数（如果为None，则使用Config中的配置）。
        batch_size: 批量提交到数据库的频道数量（默认 20）。
    """
    # 使用配置化的线程池大小
    if max_workers is None:
        max_workers = Config.get_thread_pool_size("index_build_workers", Config.CONCURRENT_PROCESSING["index_build_workers"])
    
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)

    seen: set[str] = set()

    all_ids: List[str] = []
    if seed_channel_ids:
        all_ids.extend(list(seed_channel_ids))
    if keywords:
        logger.info(f"正在通过关键词搜索频道: {', '.join(keywords)}")
        for kw in keywords:
            ids = _discover_channels_by_keyword(kw, max_results=30)  # 减少到 30 以节省配额
            if ids:
                logger.info(f"关键词 '{kw}': 找到 {len(ids)} 个频道")
            all_ids.extend(ids)

    unique_ids = [ch_id for ch_id in all_ids if ch_id not in seen]
    
    # 检查有多少频道需要更新（批量检查以提高效率）
    logger.info(f"正在检查 {len(unique_ids)} 个频道是否需要更新...")
    needs_update = []
    skipped = 0
    for ch_id in unique_ids:
        if _channel_needs_update(conn, ch_id, max_age_days=max_age_days):
            needs_update.append(ch_id)
        else:
            skipped += 1
    
    logger.info(f"频道统计: 总计 {len(unique_ids)} 个，需要更新 {len(needs_update)} 个，跳过 {skipped} 个")
    logger.info(f"配额使用估算: 约 {len(needs_update) * 101} 单位（每个频道 101 单位）")
    logger.info(f"并行处理配置: 线程数 {max_workers}，批量大小 {batch_size}")
    
    if not needs_update:
        logger.info("所有频道数据都是最新的，无需更新！")
        conn.close()
        return
    
    logger.info(f"开始并行处理 {len(needs_update)} 个需要更新的频道...")
    
    conn.close()  # 关闭主连接，工作线程会创建自己的连接
    
    processed_count = 0
    failed_count = 0
    quota_exceeded = False
    quota_used = 0
    start_time = time.time()
    
    # 批量收集处理结果
    batch_data: List[tuple] = []
    
    # 使用线程池并行处理
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
                success, channel_data = future.result()
                if success and channel_data:
                    processed_count += 1
                    quota_used += 1 + 100  # 估算配额使用
                    batch_data.append(channel_data)
                    
                    # 批量提交到数据库
                    if len(batch_data) >= batch_size:
                        _batch_upsert_channels(batch_data)
                        batch_data.clear()
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
                logger.error(f"处理频道 {ch_id} 时出错: {e}")
                continue
    
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
                cid = extract_channel_id_from_url(s, use_for="index")
            else:
                cid = s
            seed_channels.append(cid)
        except Exception as e:
            logger.warning(f"无法解析种子频道 {s}: {e}")

    # 一些默认关键词，可以按需要增删。
    default_keywords = [
        "crypto trading",
        "bitcoin",
        "ethereum",
        "criptomonedas",
        "mercado cripto",
        "defi",
        "airdrop crypto",
    ]

    build_index(
        seed_channel_ids=seed_channels, 
        keywords=default_keywords,
        max_workers=5,  # 可以根据你的网络和 API 配额调整
        batch_size=20,  # 批量提交大小
    )


