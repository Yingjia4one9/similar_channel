"""
本地数据库操作模块
处理 SQLite 数据库的读写操作
"""
import json
import os
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List

import numpy as np

from logger import get_logger

logger = get_logger()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS不可用，将使用numpy进行向量搜索")

DB_PATH = os.path.join(os.path.dirname(__file__), "channel_index.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), ".faiss_index.pkl")

# 全局索引缓存
_cached_index: Any = None
_cached_ids: List[str] | None = None
_cached_db_mtime: float | None = None

# 过期数据更新相关
_update_queue: List[str] = []
_update_queue_lock = threading.Lock()
_updating = False


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    数据库连接的 context manager，确保连接正确关闭。
    自动处理事务提交和回滚。
    
    Usage:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(...)
    
    Raises:
        sqlite3.OperationalError: 数据库操作错误（如表不存在、SQL语法错误等）
        sqlite3.IntegrityError: 数据完整性错误（如主键冲突、外键约束等）
        sqlite3.DatabaseError: 其他数据库错误
        OSError: 文件系统错误（如数据库文件无法访问）
    """
    # 从配置获取数据库超时时间（支持环境变量覆盖）
    from config import Config
    db_timeout = Config.get_config_value("DB_TIMEOUT", Config.DB_TIMEOUT, "YT_DB_TIMEOUT")
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=db_timeout)
        conn.row_factory = sqlite3.Row  # 使用 Row 工厂，便于访问列
        yield conn
        conn.commit()
    except sqlite3.OperationalError as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库操作错误（操作失败）: {e}", exc_info=True)
        raise
    except sqlite3.IntegrityError as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库完整性错误（数据冲突）: {e}", exc_info=True)
        raise
    except sqlite3.DatabaseError as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库错误: {e}", exc_info=True)
        raise
    except OSError as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库文件访问错误: {e}", exc_info=True)
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"数据库操作发生未知错误: {e}", exc_info=True)
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"关闭数据库连接时出错: {e}")


def ensure_schema() -> None:
    """
    确保数据库表结构和索引存在。
    在首次使用数据库时自动调用。
    
    Raises:
        sqlite3.DatabaseError: 如果数据库操作失败
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # 创建 channels 表（带完整性约束）
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
            
            # 创建 channel_embeddings 表（带完整性约束）
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
            # 为常用查询字段添加复合索引（如果经常一起查询）
            # 注意：SQLite会自动为主键创建索引，channel_id不需要额外索引
            
            conn.commit()
            logger.debug("数据库表结构和索引已确保存在")
    except sqlite3.DatabaseError as e:
        logger.error(f"创建数据库表结构失败: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"确保数据库表结构时发生未知错误: {e}", exc_info=True)
        raise


def _build_faiss_index() -> tuple[Any, List[str]]:
    """
    构建或加载 Faiss 索引。
    
    Returns:
        (index, channel_ids) 元组
    """
    global _cached_index, _cached_ids, _cached_db_mtime
    
    # 检查数据库是否存在
    if not os.path.exists(DB_PATH):
        logger.debug("数据库文件不存在，无法构建FAISS索引")
        return None, []
    
    # 检查数据库修改时间
    db_mtime = os.path.getmtime(DB_PATH)
    
    # 如果缓存有效，直接返回
    if _cached_index is not None and _cached_db_mtime == db_mtime:
        logger.debug(f"使用缓存的FAISS索引（包含{len(_cached_ids)}个频道）")
        return _cached_index, _cached_ids
    
    # 尝试从文件加载索引
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index_mtime = os.path.getmtime(FAISS_INDEX_PATH)
            if index_mtime >= db_mtime:
                logger.debug("从文件加载FAISS索引")
                with open(FAISS_INDEX_PATH, 'rb') as f:
                    cached_data = pickle.load(f)
                    _cached_index = cached_data['index']
                    _cached_ids = cached_data['ids']
                    _cached_db_mtime = db_mtime
                    logger.info(f"成功从文件加载FAISS索引（包含{len(_cached_ids)}个频道）")
                    return _cached_index, _cached_ids
        except FileNotFoundError:
            logger.warning("FAISS索引文件不存在，将重新构建")
        except PermissionError as e:
            logger.warning(f"无法读取FAISS索引文件（权限不足），将重新构建: {e}")
        except OSError as e:
            logger.warning(f"读取FAISS索引文件失败（文件系统错误），将重新构建: {e}")
        except Exception as e:
            logger.warning(f"从文件加载FAISS索引失败（未知错误），将重新构建: {type(e).__name__}: {e}")
    
    # 从数据库加载数据
    logger.debug("从数据库加载向量数据以构建FAISS索引")
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT channel_id, embedding FROM channel_embeddings WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库加载向量数据失败: {e}", exc_info=True)
        return None, []
    except Exception as e:
        logger.error(f"加载向量数据时发生未知错误: {e}", exc_info=True)
        return None, []
    
    if not rows:
        logger.debug("数据库中没有向量数据，无法构建FAISS索引")
        return None, []
    
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    for ch_id, blob in rows:
        if not blob:
            continue
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            ids.append(ch_id)
            vecs.append(vec)
        except (ValueError, TypeError) as e:
            logger.warning(f"解析频道 {ch_id} 的向量数据失败: {e}")
            continue
        except Exception as e:
            logger.warning(f"处理频道 {ch_id} 的向量数据时出错: {e}")
            continue
    
    if not vecs:
        return None, []
    
    mat = np.vstack(vecs).astype('float32')
    embedding_dim = mat.shape[1]
    
    if FAISS_AVAILABLE:
        # 构建 Faiss 索引（使用内积索引，向量需要归一化以实现余弦相似度）
        logger.info(f"开始构建FAISS索引（{len(ids)}个频道，向量维度：{embedding_dim}）")
        # 归一化向量
        faiss.normalize_L2(mat)
        # 使用 IndexFlatIP (内积) 进行精确搜索
        # 由于向量已归一化，内积 = 余弦相似度
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(mat)
        logger.info(f"FAISS索引构建成功（包含{len(ids)}个频道）")
    else:
        # 如果 Faiss 不可用，返回 None，将回退到原始方法
        logger.debug("FAISS不可用，将回退到numpy向量搜索")
        index = None
    
    # 缓存索引
    _cached_index = index
    _cached_ids = ids
    _cached_db_mtime = db_mtime
    
    # 保存到文件
    if index is not None:
        try:
            # 确保目录存在
            index_dir = os.path.dirname(FAISS_INDEX_PATH)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
            
            with open(FAISS_INDEX_PATH, 'wb') as f:
                pickle.dump({'index': index, 'ids': ids}, f)
            # 脱敏文件路径（CP-y5-11：敏感数据脱敏）
            from utils import sanitize_file_path
            safe_path = sanitize_file_path(FAISS_INDEX_PATH)
            logger.debug(f"FAISS索引已保存到文件: {safe_path}")
        except PermissionError as e:
            logger.warning(f"无法保存FAISS索引文件（权限不足，不影响使用）: {e}")
        except OSError as e:
            logger.warning(f"保存FAISS索引文件失败（文件系统错误，不影响使用）: {e}")
        except Exception as e:
            logger.warning(f"保存FAISS索引到文件失败（未知错误，不影响使用）: {type(e).__name__}: {e}")
    
    return index, ids


def get_candidates_from_local_index(
    base_vec: np.ndarray, max_candidates: int = 200
) -> List[str]:
    """
    从本地 SQLite 索引（build_channel_index.py 维护）中获取候选频道 ID。
    使用 Faiss 进行高效向量搜索（如果可用），否则回退到原始方法。
    若索引不存在，则返回空列表。
    
    Args:
        base_vec: 基频道的向量表示
        max_candidates: 最大返回数量
    
    Returns:
        候选频道 ID 列表
    """
    if not os.path.exists(DB_PATH):
        return []
    
    # 尝试使用 Faiss
    index, ids = _build_faiss_index()
    
    if index is not None and FAISS_AVAILABLE and ids:
        # 使用 Faiss 搜索
        logger.debug(f"使用FAISS索引搜索候选频道（最多{max_candidates}个）")
        if base_vec.ndim > 1:
            base_vec = base_vec[0]
        
        # 归一化查询向量
        query_vec = base_vec.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # 搜索 top-k
        k = min(max_candidates, len(ids))
        distances, indices = index.search(query_vec, k)
        
        # 返回对应的频道 ID
        result_ids = [ids[i] for i in indices[0] if i < len(ids)]
        logger.debug(f"FAISS搜索完成，找到{len(result_ids)}个候选频道")
        return result_ids
    
    # 回退到原始方法（如果 Faiss 不可用或构建失败）
    logger.debug("使用numpy进行向量搜索（FAISS不可用或构建失败）")
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT channel_id, embedding FROM channel_embeddings WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库加载向量数据失败（numpy回退）: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"加载向量数据时发生未知错误（numpy回退）: {e}", exc_info=True)
        return []

    if not rows:
        return []

    ids_list: List[str] = []
    vecs: List[np.ndarray] = []
    for ch_id, blob in rows:
        if not blob:
            continue
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            ids_list.append(ch_id)
            vecs.append(vec)
        except (ValueError, TypeError) as e:
            logger.warning(f"解析频道 {ch_id} 的向量数据失败: {e}")
            continue
        except Exception as e:
            logger.warning(f"处理频道 {ch_id} 的向量数据时出错: {e}")
            continue

    if not vecs:
        return []

    mat = np.vstack(vecs)
    # 计算与 base_vec 的余弦相似度，并取前 max_candidates 个频道 ID。
    if base_vec.ndim > 1:
        base_vec = base_vec[0]
    num = mat @ base_vec
    denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(base_vec) + 1e-8)
    sims = num / denom

    order = np.argsort(-sims)[:max_candidates]
    result_ids = [ids_list[i] for i in order]
    logger.debug(f"numpy向量搜索完成，找到{len(result_ids)}个候选频道")
    return result_ids


def get_channel_info_from_local_db(
    channel_ids: List[str], 
    max_age_days: int = 7,
    auto_update_expired: bool = False
) -> List[Dict[str, Any]]:
    """
    从本地数据库批量读取频道信息（用于避免配额用完时调用 API）。
    返回格式与 batch_get_channels_info 一致。
    
    优化：对于大量channel_ids，分批查询以避免IN子句过长（SQLite限制）。
    
    Args:
        channel_ids: 频道 ID 列表
        max_age_days: 数据过期时间（天），超过此时间的数据视为过期
        auto_update_expired: 是否自动更新过期数据（异步，不阻塞查询）
    
    Returns:
        频道信息列表
    """
    if not os.path.exists(DB_PATH):
        return []

    if not channel_ids:
        return []

    logger.debug(f"从本地数据库批量获取频道信息（{len(channel_ids)}个频道）")
    
    # 优化：分批查询，避免IN子句过长（SQLite的IN子句建议不超过1000个参数）
    # 使用配置的批量大小（CP-y2-15：数据库批量操作优化）
    from config import Config
    BATCH_SIZE = Config.DB_BATCH_SIZE
    all_rows = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # 分批查询
            for i in range(0, len(channel_ids), BATCH_SIZE):
                batch_ids = channel_ids[i:i + BATCH_SIZE]
                placeholders = ",".join(["?" for _ in batch_ids])
                
                # 使用EXPLAIN QUERY PLAN分析查询性能（仅在DEBUG级别）
                if logger.isEnabledFor(10):  # DEBUG级别
                    explain_plan = cur.execute(
                        f"EXPLAIN QUERY PLAN SELECT channel_id, title, description, subscriber_count, view_count, "
                        f"country, language, emails, topics, audience, updated_at "
                        f"FROM channels WHERE channel_id IN ({placeholders})",
                        batch_ids
                    ).fetchall()
                    logger.debug(f"查询计划: {explain_plan}")
                
                cur.execute(
                    f"""
                    SELECT channel_id, title, description, subscriber_count, view_count,
                           country, language, emails, topics, audience, updated_at
                    FROM channels
                    WHERE channel_id IN ({placeholders})
                    """,
                    batch_ids,
                )
                batch_rows = cur.fetchall()
                all_rows.extend(batch_rows)
            
            rows = all_rows
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库批量获取频道信息失败: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"批量获取频道信息时发生未知错误: {e}", exc_info=True)
        return []
    
    logger.debug(f"从本地数据库获取到{len(rows)}个频道的信息")
    
    # 检测过期数据（CP-y4-02：索引数据自动更新）
    # 优化：使用SQL查询过期数据，而不是在Python中遍历（更高效）
    expired_channel_ids: List[str] = []
    if auto_update_expired and rows:
        # 使用SQL直接查询过期数据（利用索引）
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                # 使用SQLite的日期函数计算过期时间
                # SQLite的日期函数：date('now', '-N days')
                cutoff_date_str = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d %H:%M:%S")
                
                # 只查询返回的频道ID中过期的
                returned_ids = [row[0] for row in rows if len(row) > 0]
                if returned_ids:
                    # 分批查询过期数据
                    for i in range(0, len(returned_ids), BATCH_SIZE):
                        batch_ids = returned_ids[i:i + BATCH_SIZE]
                        placeholders = ",".join(["?" for _ in batch_ids])
                        cur.execute(
                            f"""
                            SELECT channel_id FROM channels
                            WHERE channel_id IN ({placeholders})
                            AND updated_at < ?
                            """,
                            batch_ids + [cutoff_date_str],
                        )
                        expired_batch = [row[0] for row in cur.fetchall()]
                        expired_channel_ids.extend(expired_batch)
        except Exception as e:
            logger.warning(f"查询过期数据时出错，回退到Python检测: {e}")
            # 回退到Python检测
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            for row in rows:
                if len(row) >= 11:
                    updated_at_str = row[10]
                    if updated_at_str:
                        try:
                            if "T" in updated_at_str:
                                updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                                if updated_at.tzinfo:
                                    updated_at = updated_at.replace(tzinfo=None)
                            else:
                                updated_at = datetime.strptime(updated_at_str, "%Y-%m-%d %H:%M:%S")
                            
                            if updated_at < cutoff_date:
                                expired_channel_ids.append(row[0])
                        except Exception:
                            expired_channel_ids.append(row[0])
        
        # 异步触发过期数据更新（不阻塞查询）
        if expired_channel_ids:
            logger.info(f"检测到 {len(expired_channel_ids)} 个频道的数据已过期，将在后台更新")
            _trigger_async_update_expired_channels(expired_channel_ids)

    # 处理查询结果（注意：如果查询时包含了updated_at，需要调整索引）
    # 如果查询包含updated_at（11个字段），则调整索引；否则使用原来的10个字段
    has_updated_at = len(rows) > 0 and len(rows[0]) >= 11
    
    results: List[Dict[str, Any]] = []
    for row in rows:
        try:
            if has_updated_at:
                ch_id, title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json, _ = row
            else:
                ch_id, title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json = row
            # 安全解析JSON字段
            try:
                emails = json.loads(emails_json) if emails_json else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"解析频道 {ch_id} 的emails JSON失败: {e}")
                emails = []
            
            try:
                topics = json.loads(topics_json) if topics_json else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"解析频道 {ch_id} 的topics JSON失败: {e}")
                topics = []
            
            try:
                audience = json.loads(audience_json) if audience_json else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"解析频道 {ch_id} 的audience JSON失败: {e}")
                audience = []
            
            results.append(
                {
                    "channelId": ch_id,
                    "title": title or "",
                    "description": desc or "",
                    "thumbnails": {},  # 本地库暂不存缩略图
                    "subscriberCount": int(sub_count or 0) if sub_count is not None else 0,
                    "videoCount": 0,  # 本地库暂不存视频数
                    "viewCount": int(view_count or 0) if view_count is not None else 0,
                    "defaultLanguage": lang,
                    "country": country,
                    "emails": emails,
                    "topics": topics,
                    "audience": audience,
                    "recent_videos": [],  # 本地库暂不存最近视频，可后续扩展
                }
            )
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"处理数据库行数据时出错: {e}, row: {row}")
            continue
        except Exception as e:
            logger.warning(f"处理频道数据时发生未知错误: {e}", exc_info=True)
            continue
    return results


def get_single_channel_info_from_local_db(channel_id: str) -> Dict[str, Any] | None:
    """
    从本地数据库读取单个频道的完整信息（用于配额用尽时的降级策略）。
    返回格式与 get_channel_basic_info 一致。
    
    Args:
        channel_id: 频道 ID
    
    Returns:
        频道信息字典，如果不存在则返回 None
    """
    if not os.path.exists(DB_PATH):
        return None

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT channel_id, title, description, subscriber_count, view_count,
                       country, language, emails, topics, audience
                FROM channels
                WHERE channel_id = ?
                """,
                (channel_id,),
            )
            row = cur.fetchone()
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库获取频道信息失败 (channel_id={channel_id}): {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"获取频道信息时发生未知错误 (channel_id={channel_id}): {e}", exc_info=True)
        return None

    if not row:
        return None

    try:
        ch_id, title, desc, sub_count, view_count, country, lang, emails_json, topics_json, audience_json = row
        
        # 安全解析JSON字段
        try:
            emails = json.loads(emails_json) if emails_json else []
        except (json.JSONDecodeError, TypeError):
            emails = []
        
        try:
            topics = json.loads(topics_json) if topics_json else []
        except (json.JSONDecodeError, TypeError):
            topics = []
        
        try:
            audience = json.loads(audience_json) if audience_json else []
        except (json.JSONDecodeError, TypeError):
            audience = []
        
        return {
            "channelId": ch_id,
            "title": title or "",
            "description": desc or "",
            "thumbnails": {},  # 本地库暂不存缩略图
            "subscriberCount": int(sub_count or 0) if sub_count is not None else 0,
            "videoCount": 0,  # 本地库暂不存视频数
            "viewCount": int(view_count or 0) if view_count is not None else 0,
            "defaultLanguage": lang,
            "defaultAudioLanguage": None,  # 本地库暂不存音频语言
            "country": country,
            "emails": emails,
            "topics": topics,
            "audience": audience,
        }
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"解析频道数据失败 (channel_id={channel_id}): {e}", exc_info=True)
        return None


def get_channel_basic_info_for_filtering(channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    从本地数据库批量读取频道基础信息（用于粗筛）。
    只返回订阅数、语言、国家等基本字段。
    
    优化：分批查询以避免IN子句过长。
    
    Args:
        channel_ids: 频道 ID 列表
    
    Returns:
        字典，键为 channel_id，值为包含基础信息的字典
    """
    if not os.path.exists(DB_PATH):
        return {}

    if not channel_ids:
        return {}

    # 优化：分批查询
    # 使用配置的批量大小（CP-y2-15：数据库批量操作优化）
    from config import Config
    BATCH_SIZE = Config.DB_BATCH_SIZE
    all_rows = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            for i in range(0, len(channel_ids), BATCH_SIZE):
                batch_ids = channel_ids[i:i + BATCH_SIZE]
                placeholders = ",".join(["?" for _ in batch_ids])
                cur.execute(
                    f"""
                    SELECT channel_id, subscriber_count, language, country
                    FROM channels
                    WHERE channel_id IN ({placeholders})
                    """,
                    batch_ids,
                )
                batch_rows = cur.fetchall()
                all_rows.extend(batch_rows)
            
            rows = all_rows
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库批量获取频道基础信息失败: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"批量获取频道基础信息时发生未知错误: {e}", exc_info=True)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            ch_id, sub_count, lang, country = row
            result[ch_id] = {
                "subscriberCount": int(sub_count or 0) if sub_count is not None else 0,
                "defaultLanguage": lang,
                "country": country,
            }
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"处理频道基础信息行数据时出错: {e}, row: {row}")
            continue
    return result


def get_embeddings_from_local_db(channel_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    从本地数据库批量读取频道向量。
    
    优化：分批查询以避免IN子句过长。
    
    Args:
        channel_ids: 频道 ID 列表
    
    Returns:
        字典，键为 channel_id，值为向量数组
    """
    if not os.path.exists(DB_PATH):
        return {}

    if not channel_ids:
        return {}

    # 优化：分批查询
    # 使用配置的批量大小（CP-y2-15：数据库批量操作优化）
    from config import Config
    BATCH_SIZE = Config.DB_BATCH_SIZE
    all_rows = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            for i in range(0, len(channel_ids), BATCH_SIZE):
                batch_ids = channel_ids[i:i + BATCH_SIZE]
                placeholders = ",".join(["?" for _ in batch_ids])
                cur.execute(
                    f"""
                    SELECT channel_id, embedding
                    FROM channel_embeddings
                    WHERE channel_id IN ({placeholders})
                    """,
                    batch_ids,
                )
                batch_rows = cur.fetchall()
                all_rows.extend(batch_rows)
            
            rows = all_rows
    except sqlite3.DatabaseError as e:
        logger.error(f"从数据库批量获取频道向量失败: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"批量获取频道向量时发生未知错误: {e}", exc_info=True)
        return {}

    result: Dict[str, np.ndarray] = {}
    for ch_id, blob in rows:
        if blob:
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
                result[ch_id] = vec
            except (ValueError, TypeError) as e:
                logger.warning(f"解析频道 {ch_id} 的向量数据失败: {e}")
                continue
            except Exception as e:
                logger.warning(f"处理频道 {ch_id} 的向量数据时出错: {e}")
                continue
    return result


def _trigger_async_update_expired_channels(channel_ids: List[str]) -> None:
    """
    异步触发过期频道数据的更新（CP-y4-02：索引数据自动更新）
    
    将需要更新的频道ID添加到更新队列，在后台线程中处理。
    不阻塞当前查询，确保用户体验。
    
    Args:
        channel_ids: 需要更新的频道ID列表
    """
    global _update_queue, _updating
    
    if not channel_ids:
        return
    
    with _update_queue_lock:
        # 添加到更新队列（去重）
        for cid in channel_ids:
            if cid and cid not in _update_queue:
                _update_queue.append(cid)
        
        # 如果还没有更新线程在运行，启动一个
        if not _updating and _update_queue:
            _updating = True
            update_thread = threading.Thread(
                target=_process_update_queue,
                daemon=True,
                name="ChannelDataUpdater"
            )
            update_thread.start()
            logger.debug(f"启动后台更新线程，待更新频道数: {len(_update_queue)}")


def _process_update_queue() -> None:
    """
    处理更新队列，批量更新过期频道数据
    
    在后台线程中运行，不阻塞主流程。
    使用build_channel_index模块的功能来更新数据。
    """
    global _update_queue, _updating
    
    try:
        # 延迟导入，避免循环依赖
        from build_channel_index import _process_channel_data, _batch_upsert_channels
        from cache import invalidate_all_channel_caches
        
        while True:
            # 从队列中取出需要更新的频道ID
            with _update_queue_lock:
                if not _update_queue:
                    _updating = False
                    break
                
                # 每次处理最多10个频道，避免一次性处理太多
                batch_size = min(10, len(_update_queue))
                batch_ids = _update_queue[:batch_size]
                _update_queue = _update_queue[batch_size:]
            
            if not batch_ids:
                continue
            
            logger.info(f"开始后台更新 {len(batch_ids)} 个过期频道的数据")
            
            # 处理这批频道
            channels_to_update = []
            for channel_id in batch_ids:
                try:
                    # 处理频道数据（获取信息、计算向量）
                    info, vec = _process_channel_data(channel_id, recent_videos_count=3)
                    if info and vec is not None:
                        # 准备数据用于数据库更新
                        from youtube_client import _prepare_channel_data_for_db
                        channel_data = _prepare_channel_data_for_db(info, vec)
                        if channel_data:
                            channels_to_update.append(channel_data)
                except Exception as e:
                    logger.warning(f"后台更新频道 {channel_id} 失败: {e}")
                    continue
            
            # 批量更新到数据库
            if channels_to_update:
                try:
                    _batch_upsert_channels(channels_to_update)
                    logger.info(f"成功后台更新 {len(channels_to_update)} 个频道的数据")
                    
                    # 失效相关缓存
                    for channel_data in channels_to_update:
                        if channel_data and len(channel_data) > 0:
                            channel_id = channel_data[0]
                            invalidate_all_channel_caches(channel_id)
                except Exception as e:
                    logger.warning(f"批量更新过期频道数据失败: {e}", exc_info=True)
            
            # 短暂休眠，避免占用过多资源
            import time
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"处理更新队列时发生错误: {e}", exc_info=True)
    finally:
        with _update_queue_lock:
            _updating = False
            if _update_queue:
                # 如果还有待更新的频道，重新启动更新线程
                _updating = True
                update_thread = threading.Thread(
                    target=_process_update_queue,
                    daemon=True,
                    name="ChannelDataUpdater"
                )
                update_thread.start()


def get_expired_channel_ids(max_age_days: int = 7) -> List[str]:
    """
    获取所有过期频道的ID列表（CP-y4-02：索引数据自动更新）
    
    Args:
        max_age_days: 数据过期时间（天）
    
    Returns:
        过期频道的ID列表
    """
    if not os.path.exists(DB_PATH):
        return []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            cur.execute(
                """
                SELECT channel_id FROM channels
                WHERE updated_at < ? OR updated_at IS NULL
                """,
                (cutoff_date_str,),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows if row and row[0]]
    except Exception as e:
        logger.error(f"查询过期频道失败: {e}", exc_info=True)
        return []


def check_vector_info_consistency(channel_ids: List[str] | None = None) -> Dict[str, bool]:
    """
    检查向量与信息的一致性（CP-y4-09：向量与信息一致性）
    
    检查指定频道（或所有频道）的向量是否存在且与信息同步。
    如果频道信息存在但向量不存在，则返回不一致。
    
    Args:
        channel_ids: 要检查的频道ID列表，如果为None则检查所有频道
    
    Returns:
        字典，键为channel_id，值为True（一致）或False（不一致）
    """
    if not os.path.exists(DB_PATH):
        return {}
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            if channel_ids:
                placeholders = ",".join(["?" for _ in channel_ids])
                cur.execute(
                    f"""
                    SELECT c.channel_id, e.channel_id as has_embedding
                    FROM channels c
                    LEFT JOIN channel_embeddings e ON c.channel_id = e.channel_id
                    WHERE c.channel_id IN ({placeholders})
                    """,
                    channel_ids,
                )
            else:
                cur.execute(
                    """
                    SELECT c.channel_id, e.channel_id as has_embedding
                    FROM channels c
                    LEFT JOIN channel_embeddings e ON c.channel_id = e.channel_id
                    """
                )
            
            rows = cur.fetchall()
            
            result: Dict[str, bool] = {}
            for row in rows:
                if len(row) >= 2:
                    ch_id = row[0]
                    has_embedding = row[1] is not None
                    # 如果频道信息存在但向量不存在，则不一致
                    result[ch_id] = has_embedding
            
            return result
    except Exception as e:
        logger.error(f"检查向量与信息一致性失败: {e}", exc_info=True)
        return {}


def recalculate_vectors_for_channels(channel_ids: List[str]) -> int:
    """
    为指定频道重新计算向量（CP-y4-09：向量与信息一致性）
    
    当频道信息更新但向量未更新时，调用此函数重新计算向量。
    
    Args:
        channel_ids: 需要重新计算向量的频道ID列表
    
    Returns:
        成功重新计算的频道数量
    """
    if not channel_ids:
        return 0
    
    try:
        # 延迟导入，避免循环依赖
        from build_channel_index import _process_channel_data, _batch_upsert_channels
        from cache import invalidate_all_channel_caches
        
        logger.info(f"开始为 {len(channel_ids)} 个频道重新计算向量")
        
        channels_to_update = []
        for channel_id in channel_ids:
            try:
                # 处理频道数据（获取信息、计算向量）
                info, vec = _process_channel_data(channel_id, recent_videos_count=3)
                if info and vec is not None:
                    # 准备数据用于数据库更新
                    from youtube_client import _prepare_channel_data_for_db
                    channel_data = _prepare_channel_data_for_db(info, vec)
                    if channel_data:
                        channels_to_update.append(channel_data)
            except Exception as e:
                logger.warning(f"重新计算频道 {channel_id} 的向量失败: {e}")
                continue
        
        # 批量更新到数据库
        if channels_to_update:
            try:
                _batch_upsert_channels(channels_to_update)
                logger.info(f"成功为 {len(channels_to_update)} 个频道重新计算向量")
                
                # 失效相关缓存
                for channel_data in channels_to_update:
                    if channel_data and len(channel_data) > 0:
                        channel_id = channel_data[0]
                        invalidate_all_channel_caches(channel_id)
                
                return len(channels_to_update)
            except Exception as e:
                logger.error(f"批量更新向量失败: {e}", exc_info=True)
                return 0
        
        return 0
    except Exception as e:
        logger.error(f"重新计算向量时发生错误: {e}", exc_info=True)
        return 0
