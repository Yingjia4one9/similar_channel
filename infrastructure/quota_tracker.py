"""
YouTube API 配额跟踪模块
跟踪每次 API 调用的配额消耗，提供配额使用统计和查询功能
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.logger import get_logger
from infrastructure.config import Config

logger = get_logger()

# 配额跟踪数据库路径
QUOTA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "quota_tracker.db"))

# YouTube API 配额消耗表（根据官方文档）
# 注意：这些值可能随API版本更新而变化，需要定期检查
QUOTA_COSTS = {
    "search": {
        "list": 100,  # 搜索请求
    },
    "channels": {
        "list": 1,  # 频道信息查询
    },
    "videos": {
        "list": 1,  # 视频信息查询
    },
    "playlistItems": {
        "list": 1,  # 播放列表项查询
    },
}

# 默认每日配额（免费配额）
DEFAULT_DAILY_QUOTA = 10000

# 配额告警阈值（使用率超过此值时告警）
QUOTA_WARNING_THRESHOLD = 0.8  # 80%

# 线程安全的锁
_quota_lock = threading.Lock()

# 配额限流状态（CP-y3-05：配额限流机制）
# 按 use_for 分别存储限流状态
_rate_limit_status: Dict[str, Dict[str, Any]] = {}  # {use_for: {"enabled": bool, "delay": float}}
_rate_limit_lock = threading.Lock()

# 告警检查计数器（每N次配额记录后检查一次告警，避免频繁检查）
_quota_record_count = 0
_quota_record_count_lock = threading.Lock()
_ALERT_CHECK_INTERVAL = 10  # 每10次配额记录检查一次告警


def _ensure_quota_schema(conn: sqlite3.Connection) -> None:
    """确保配额跟踪表结构存在"""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quota_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            cost INTEGER NOT NULL,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            params_json TEXT,
            success INTEGER NOT NULL DEFAULT 1,
            use_for TEXT
        )
        """
    )
    # 如果表已存在但没有 use_for 字段，添加该字段（数据库迁移）
    try:
        cur.execute("ALTER TABLE quota_usage ADD COLUMN use_for TEXT")
    except sqlite3.OperationalError:
        # 字段已存在，忽略错误
        pass
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quota_fallback_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fallback_type TEXT NOT NULL,
            endpoint TEXT,
            context TEXT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            success INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_quota_timestamp 
        ON quota_usage(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_quota_use_for 
        ON quota_usage(use_for)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_quota_timestamp_use_for 
        ON quota_usage(timestamp, use_for)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fallback_timestamp 
        ON quota_fallback_stats(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fallback_type 
        ON quota_fallback_stats(fallback_type)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quota_warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            warning_type TEXT NOT NULL,
            usage_rate REAL NOT NULL,
            used_quota INTEGER NOT NULL,
            total_quota INTEGER NOT NULL,
            message TEXT,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            acknowledged INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_warning_timestamp 
        ON quota_warnings(timestamp)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_warning_acknowledged 
        ON quota_warnings(acknowledged)
        """
    )
    conn.commit()


@contextmanager
def get_quota_db_connection():
    """配额数据库连接的 context manager"""
    conn = sqlite3.connect(QUOTA_DB_PATH)
    try:
        _ensure_quota_schema(conn)
        yield conn
    finally:
        conn.close()


def get_quota_cost(endpoint: str, method: str = "list") -> int:
    """
    获取指定 API 调用的配额消耗
    
    Args:
        endpoint: API 端点（如 "search", "channels"）
        method: 方法名（通常是 "list"）
    
    Returns:
        配额消耗量（单位），如果未知则返回默认值 1
    """
    return QUOTA_COSTS.get(endpoint, {}).get(method, 1)


def record_quota_usage(
    endpoint: str,
    method: str = "list",
    cost: Optional[int] = None,
    params: Optional[Dict] = None,
    success: bool = True,
    use_for: Optional[str] = None,
) -> None:
    """
    记录配额使用情况
    
    Args:
        endpoint: API 端点
        method: 方法名
        cost: 配额消耗量，如果为 None 则自动计算
        params: 请求参数（用于调试）
        success: 是否成功（用于统计成功率）
        use_for: API Key 用途标识（"index" 或 "search"），None 表示默认/未指定
    """
    if cost is None:
        cost = get_quota_cost(endpoint, method)
    
    params_json = json.dumps(params) if params else None
    timestamp = datetime.now(timezone.utc).isoformat()
    
    try:
        with _quota_lock:
            with get_quota_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO quota_usage (endpoint, method, cost, timestamp, params_json, success, use_for)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (endpoint, method, cost, timestamp, params_json, 1 if success else 0, use_for),
                )
                conn.commit()
        
        # 定期检查配额告警和限流状态（每N次记录后检查一次，避免频繁检查）
        # 修复：使用原子操作避免竞态条件
        global _quota_record_count
        should_check = False
        with _quota_record_count_lock:
            _quota_record_count += 1
            if _quota_record_count >= _ALERT_CHECK_INTERVAL:
                _quota_record_count = 0
                should_check = True
        
        # 在锁外进行检查，但使用原子操作确保不会重复检查
        if should_check:
            log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "location": "quota_tracker.py:record_quota_usage",
                        "message": "开始检查配额告警和限流（修复异步检查竞态）",
                        "data": {"endpoint": endpoint, "use_for": use_for},
                        "sessionId": "debug-session",
                        "runId": "fix-verification",
                        "hypothesisId": "D"
                    }, ensure_ascii=False) + "\n")
            except: pass
            # #endregion
            
            try:
                # 检查所有用途的配额告警和限流
                # 优化：避免在锁内执行数据库操作，防止阻塞
                for usage_type in [None, "index", "search"]:
                    # 先快速获取使用率（在锁外，避免阻塞）
                    usage_rate = get_quota_usage_rate(daily_quota=DEFAULT_DAILY_QUOTA, use_for=usage_type)
                    # 快速检查限流状态（使用已获取的使用率，避免在锁内查询数据库）
                    with _rate_limit_lock:
                        # 使用已获取的使用率，避免在锁内查询数据库
                        use_for_key = usage_type if usage_type is not None else "default"
                        usage_rate_decimal = usage_rate / 100.0
                        threshold = Config.QUOTA_RATE_LIMIT.get("threshold", 0.8)
                        strict_threshold = Config.QUOTA_RATE_LIMIT.get("strict_threshold", 0.95)
                        reduction_rate = Config.QUOTA_RATE_LIMIT.get("reduction_rate", 0.5)
                        min_delay = Config.QUOTA_RATE_LIMIT.get("min_delay_seconds", 0.1)
                        max_delay = Config.QUOTA_RATE_LIMIT.get("max_delay_seconds", 5.0)
                        
                        if use_for_key not in _rate_limit_status:
                            _rate_limit_status[use_for_key] = {"enabled": False, "delay": 0.0}
                        
                        if usage_rate_decimal >= strict_threshold:
                            _rate_limit_status[use_for_key]["enabled"] = True
                            _rate_limit_status[use_for_key]["delay"] = max_delay
                        elif usage_rate_decimal >= threshold:
                            _rate_limit_status[use_for_key]["enabled"] = True
                            excess_rate = (usage_rate_decimal - threshold) / (strict_threshold - threshold)
                            base_delay = min_delay * (1 / reduction_rate)
                            calculated_delay = base_delay * (1 + excess_rate * (max_delay / base_delay - 1))
                            delay = min(max(calculated_delay, min_delay), max_delay)
                            _rate_limit_status[use_for_key]["delay"] = delay
                        else:
                            if _rate_limit_status[use_for_key]["enabled"]:
                                pass  # 取消限流的日志已在原函数中
                            _rate_limit_status[use_for_key]["enabled"] = False
                            _rate_limit_status[use_for_key]["delay"] = 0.0
                    
                    # 告警检查涉及数据库操作，在锁外执行，避免阻塞
                    check_and_record_quota_warning(daily_quota=DEFAULT_DAILY_QUOTA, use_for=usage_type)
                
                # #region agent log
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "timestamp": int(time.time() * 1000),
                            "location": "quota_tracker.py:record_quota_usage",
                            "message": "配额告警和限流检查完成",
                            "data": {"endpoint": endpoint, "use_for": use_for},
                            "sessionId": "debug-session",
                            "runId": "fix-verification",
                            "hypothesisId": "D"
                        }, ensure_ascii=False) + "\n")
                except: pass
                # #endregion
            except Exception as e:
                logger.debug(f"检查配额告警和限流时出错: {e}")
                # #region agent log
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "timestamp": int(time.time() * 1000),
                            "location": "quota_tracker.py:record_quota_usage",
                            "message": "配额告警和限流检查出错",
                            "data": {"error": str(e), "endpoint": endpoint, "use_for": use_for},
                            "sessionId": "debug-session",
                            "runId": "fix-verification",
                            "hypothesisId": "D"
                        }, ensure_ascii=False) + "\n")
                except: pass
                # #endregion
    except Exception as e:
        logger.warning(f"记录配额使用失败: {e}")


def get_quota_usage_today(daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> Dict:
    """
    获取今天的配额使用情况
    
    Args:
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有或默认
    
    Returns:
        包含以下字段的字典：
        - used: 已使用配额
        - total: 总配额
        - remaining: 剩余配额
        - usage_rate: 使用率（0-100）
        - count: API 调用次数
        - success_count: 成功次数
        - fail_count: 失败次数
        - use_for: 用途标识
    """
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            if use_for is not None:
                cur.execute(
                    """
                    SELECT 
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count,
                        COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count
                    FROM quota_usage
                    WHERE timestamp >= ? AND use_for = ?
                    """,
                    (today_start, use_for),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count,
                        COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count
                    FROM quota_usage
                    WHERE timestamp >= ? AND (use_for IS NULL OR use_for = '')
                    """,
                    (today_start,),
                )
            row = cur.fetchone()
            used = int(row[0] or 0)
            count = int(row[1] or 0)
            success_count = int(row[2] or 0)
    except Exception as e:
        logger.warning(f"查询配额使用情况失败: {e}")
        used = 0
        count = 0
        success_count = 0
    
    remaining = max(0, daily_quota - used)
    usage_rate = (used / daily_quota * 100) if daily_quota > 0 else 0
    fail_count = count - success_count
    
    return {
        "used": used,
        "total": daily_quota,
        "remaining": remaining,
        "usage_rate": round(usage_rate, 2),
        "count": count,
        "success_count": success_count,
        "fail_count": fail_count,
        "use_for": use_for,
    }


def get_quota_usage_logs(
    days: int = 1,
    limit: int = 200,
    success: Optional[bool] = None,
    use_for: Optional[str] = None,
) -> Dict[str, Any]:
    """
    获取配额使用日志（CP-y3-13：配额使用日志）
    
    Args:
        days: 查询最近N天的日志
        limit: 返回的最大条数
        success: 是否只查询成功/失败的请求，None 表示全部
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有
    
    Returns:
        包含日志列表和统计信息的字典
    """
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(day=start_date.day - days + 1).isoformat()
    
    logs: List[Dict[str, Any]] = []
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            query = """
                SELECT endpoint, method, cost, timestamp, params_json, success, use_for
                FROM quota_usage
                WHERE timestamp >= ?
            """
            params: List[Any] = [start_date]
            if use_for is not None:
                query += " AND use_for = ?"
                params.append(use_for)
            else:
                # 如果 use_for 为 None，查询所有（包括 NULL 和空字符串）
                query += " AND (use_for IS NULL OR use_for = '')"
            if success is not None:
                query += " AND success = ?"
                params.append(1 if success else 0)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            for row in cur.execute(query, params):
                endpoint, method, cost, ts, params_json, succ, use_for_val = row
                logs.append(
                    {
                        "endpoint": endpoint,
                        "method": method,
                        "cost": cost,
                        "timestamp": ts,
                        "params": json.loads(params_json) if params_json else None,
                        "success": bool(succ),
                        "use_for": use_for_val,
                    }
                )
    except Exception as e:
        logger.warning(f"查询配额使用日志失败: {e}")
        return {"logs": [], "count": 0}
    
    return {"logs": logs, "count": len(logs)}


def get_quota_usage_by_endpoint(days: int = 1, use_for: Optional[str] = None) -> Dict[str, Dict]:
    """
    按端点统计配额使用情况
    
    Args:
        days: 统计最近N天的数据
        use_for: API Key 用途标识（"index" 或 "search"），None 表示统计所有
    
    Returns:
        字典，键为 endpoint，值为包含 cost 和 count 的字典
    """
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(day=start_date.day - days + 1).isoformat()
    
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            if use_for is not None:
                cur.execute(
                    """
                    SELECT 
                        endpoint,
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count
                    FROM quota_usage
                    WHERE timestamp >= ? AND use_for = ?
                    GROUP BY endpoint
                    """,
                    (start_date, use_for),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        endpoint,
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count
                    FROM quota_usage
                    WHERE timestamp >= ? AND (use_for IS NULL OR use_for = '')
                    GROUP BY endpoint
                    """,
                    (start_date,),
                )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning(f"按端点统计配额使用失败: {e}")
        return {}
    
    result = {}
    for endpoint, cost, count in rows:
        result[endpoint] = {
            "cost": int(cost),
            "count": int(count),
        }
    
    return result


def record_fallback_usage(
    fallback_type: str,
    endpoint: str | None = None,
    context: str | None = None,
    success: bool = True,
) -> None:
    """
    记录配额耗尽时的降级使用情况
    
    Args:
        fallback_type: 降级类型（如 "local_db_channel_info", "local_db_candidates", "skip_operation"）
        endpoint: 原始API端点（如果适用）
        context: 上下文信息（如频道ID、操作描述等）
        success: 降级是否成功
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    try:
        with _quota_lock:
            with get_quota_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO quota_fallback_stats (fallback_type, endpoint, context, timestamp, success)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (fallback_type, endpoint, context, timestamp, 1 if success else 0),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"记录降级使用失败: {e}")


def get_fallback_stats(days: int = 1) -> Dict:
    """
    获取降级统计信息
    
    Args:
        days: 统计最近N天的数据
    
    Returns:
        包含以下字段的字典：
        - total_fallbacks: 总降级次数
        - success_count: 成功降级次数
        - fail_count: 失败降级次数
        - by_type: 按降级类型统计的字典
        - by_endpoint: 按端点统计的字典
    """
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(day=start_date.day - days + 1).isoformat()
    
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            
            # 总体统计
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_count,
                    COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count
                FROM quota_fallback_stats
                WHERE timestamp >= ?
                """,
                (start_date,),
            )
            row = cur.fetchone()
            total_count = int(row[0] or 0)
            success_count = int(row[1] or 0)
            fail_count = total_count - success_count
            
            # 按类型统计
            cur.execute(
                """
                SELECT 
                    fallback_type,
                    COUNT(*) as count,
                    COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as success_count
                FROM quota_fallback_stats
                WHERE timestamp >= ?
                GROUP BY fallback_type
                """,
                (start_date,),
            )
            by_type = {}
            for fallback_type, count, success in cur.fetchall():
                by_type[fallback_type] = {
                    "count": int(count),
                    "success_count": int(success),
                    "fail_count": int(count) - int(success),
                }
            
            # 按端点统计
            cur.execute(
                """
                SELECT 
                    endpoint,
                    COUNT(*) as count
                FROM quota_fallback_stats
                WHERE timestamp >= ? AND endpoint IS NOT NULL
                GROUP BY endpoint
                """,
                (start_date,),
            )
            by_endpoint = {}
            for endpoint, count in cur.fetchall():
                by_endpoint[endpoint] = int(count)
            
            return {
                "total_fallbacks": total_count,
                "success_count": success_count,
                "fail_count": fail_count,
                "by_type": by_type,
                "by_endpoint": by_endpoint,
            }
    except Exception as e:
        logger.warning(f"查询降级统计失败: {e}")
        return {
            "total_fallbacks": 0,
            "success_count": 0,
            "fail_count": 0,
            "by_type": {},
            "by_endpoint": {},
        }


def reset_quota_stats() -> None:
    """
    清理旧的配额统计数据（保留最近30天的数据）
    """
    cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_date = cutoff_date.replace(day=cutoff_date.day - 30).isoformat()
    
    try:
        with _quota_lock:
            with get_quota_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM quota_usage WHERE timestamp < ?",
                    (cutoff_date,),
                )
                deleted_usage = cur.rowcount
                cur.execute(
                    "DELETE FROM quota_fallback_stats WHERE timestamp < ?",
                    (cutoff_date,),
                )
                deleted_fallback = cur.rowcount
                conn.commit()
                logger.info(f"清理了 {deleted_usage} 条旧的配额使用记录和 {deleted_fallback} 条降级统计记录")
    except Exception as e:
        logger.warning(f"清理配额统计数据失败: {e}")


def clean_legacy_quota_records(dry_run: bool = False) -> Dict[str, int]:
    """
    清理添加 use_for 字段之前的旧记录（use_for IS NULL 的记录）
    
    这些旧记录是在添加多 API Key 支持之前创建的，会影响默认用途的配额统计。
    清理后，默认用途（use_for IS NULL）的统计将只包含真正使用默认 API Key 的记录。
    
    Args:
        dry_run: 如果为 True，只统计要删除的记录数，不实际删除
    
    Returns:
        包含清理统计信息的字典：
        - deleted_count: 删除的记录数
        - total_count: 旧记录总数
        - dry_run: 是否为试运行
    """
    try:
        with _quota_lock:
            with get_quota_db_connection() as conn:
                cur = conn.cursor()
                
                # 先统计要删除的记录数
                cur.execute(
                    """
                    SELECT COUNT(*) FROM quota_usage
                    WHERE use_for IS NULL
                    """
                )
                total_count = cur.fetchone()[0]
                
                if dry_run:
                    logger.info(f"[试运行] 发现 {total_count} 条旧记录（use_for IS NULL）待清理")
                    return {
                        "deleted_count": 0,
                        "total_count": total_count,
                        "dry_run": True,
                    }
                
                # 执行删除
                cur.execute(
                    """
                    DELETE FROM quota_usage
                    WHERE use_for IS NULL
                    """
                )
                deleted_count = cur.rowcount
                conn.commit()
                
                logger.info(f"清理了 {deleted_count} 条旧记录（use_for IS NULL）")
                
                return {
                    "deleted_count": deleted_count,
                    "total_count": total_count,
                    "dry_run": False,
                }
    except Exception as e:
        logger.warning(f"清理旧记录失败: {e}")
        return {
            "deleted_count": 0,
            "total_count": 0,
            "dry_run": dry_run,
            "error": str(e),
        }


def get_quota_usage_rate(daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> float:
    """
    计算配额使用率（0-100）
    
    Args:
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有或默认
    
    Returns:
        使用率（0-100），如果配额为0则返回0
    """
    usage_info = get_quota_usage_today(daily_quota, use_for)
    return usage_info.get("usage_rate", 0.0)


def is_quota_exceeded(daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> bool:
    """
    检查配额是否已耗尽
    
    Args:
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有或默认
    
    Returns:
        True 如果配额已耗尽，False 否则
    """
    usage_info = get_quota_usage_today(daily_quota, use_for)
    return usage_info.get("remaining", 0) <= 0


def check_and_update_rate_limit(daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> tuple[bool, float]:
    """
    检查并更新配额限流状态（CP-y3-05：配额限流机制）
    
    Args:
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示默认
    
    Returns:
        (is_rate_limited, delay_seconds) 元组
        - is_rate_limited: 是否处于限流状态
        - delay_seconds: 建议的延迟时间（秒）
    """
    if not Config.QUOTA_RATE_LIMIT.get("enabled", True):
        return (False, 0.0)
    
    # 使用 None 作为默认用途的键
    use_for_key = use_for if use_for is not None else "default"
    
    usage_rate = get_quota_usage_rate(daily_quota, use_for)
    # usage_rate 是 0-100 的百分比，需要转换为 0-1 的小数进行比较
    usage_rate_decimal = usage_rate / 100.0
    threshold = Config.QUOTA_RATE_LIMIT.get("threshold", 0.8)
    strict_threshold = Config.QUOTA_RATE_LIMIT.get("strict_threshold", 0.95)
    reduction_rate = Config.QUOTA_RATE_LIMIT.get("reduction_rate", 0.5)
    min_delay = Config.QUOTA_RATE_LIMIT.get("min_delay_seconds", 0.1)
    max_delay = Config.QUOTA_RATE_LIMIT.get("max_delay_seconds", 5.0)
    
    global _rate_limit_status
    
    with _rate_limit_lock:
        if use_for_key not in _rate_limit_status:
            _rate_limit_status[use_for_key] = {"enabled": False, "delay": 0.0}
        
        if usage_rate_decimal >= strict_threshold:
            # 严格限流：使用最大延迟
            _rate_limit_status[use_for_key]["enabled"] = True
            _rate_limit_status[use_for_key]["delay"] = max_delay
            logger.warning(
                f"[{use_for_key}] 配额使用率 {usage_rate:.1f}% 超过严格限流阈值 {strict_threshold*100:.0f}%，"
                f"启用严格限流（延迟 {max_delay} 秒）"
            )
        elif usage_rate_decimal >= threshold:
            # 普通限流：根据使用率计算延迟
            _rate_limit_status[use_for_key]["enabled"] = True
            # 延迟 = 基础延迟 * (使用率 - 阈值) / (严格阈值 - 阈值) * (1 / reduction_rate)
            excess_rate = (usage_rate_decimal - threshold) / (strict_threshold - threshold)
            base_delay = min_delay * (1 / reduction_rate)
            calculated_delay = base_delay * (1 + excess_rate * (max_delay / base_delay - 1))
            delay = min(max(calculated_delay, min_delay), max_delay)
            _rate_limit_status[use_for_key]["delay"] = delay
            logger.info(
                f"[{use_for_key}] 配额使用率 {usage_rate:.1f}% 超过限流阈值 {threshold*100:.0f}%，"
                f"启用限流（延迟 {delay:.2f} 秒）"
            )
        else:
            # 取消限流
            if _rate_limit_status[use_for_key]["enabled"]:
                logger.info(f"[{use_for_key}] 配额使用率 {usage_rate:.1f}% 低于限流阈值，取消限流")
            _rate_limit_status[use_for_key]["enabled"] = False
            _rate_limit_status[use_for_key]["delay"] = 0.0
    
    status = _rate_limit_status[use_for_key]
    return (status["enabled"], status["delay"])


def get_rate_limit_status(use_for: Optional[str] = None) -> Dict:
    """
    获取当前配额限流状态（CP-y3-05：配额限流机制）
    
    Args:
        use_for: API Key 用途标识（"index" 或 "search"），None 表示默认
    
    Returns:
        包含限流状态的字典
    """
    use_for_key = use_for if use_for is not None else "default"
    
    with _rate_limit_lock:
        if use_for_key not in _rate_limit_status:
            _rate_limit_status[use_for_key] = {"enabled": False, "delay": 0.0}
        
        status = _rate_limit_status[use_for_key]
        return {
            "enabled": status["enabled"],
            "delay_seconds": status["delay"],
            "use_for": use_for,
            "config": {
                "threshold": Config.QUOTA_RATE_LIMIT.get("threshold", 0.8),
                "strict_threshold": Config.QUOTA_RATE_LIMIT.get("strict_threshold", 0.95),
                "reduction_rate": Config.QUOTA_RATE_LIMIT.get("reduction_rate", 0.5),
            }
        }


def get_quota_status(daily_quota: int = DEFAULT_DAILY_QUOTA, warning_threshold: float = QUOTA_WARNING_THRESHOLD, use_for: Optional[str] = None) -> Dict:
    """
    获取配额状态信息
    
    Args:
        daily_quota: 每日配额上限
        warning_threshold: 告警阈值（0-1），默认0.8（80%）
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有或默认
    
    Returns:
        包含以下字段的字典：
        - used: 已使用配额
        - total: 总配额
        - remaining: 剩余配额
        - usage_rate: 使用率（0-100）
        - is_exceeded: 是否已耗尽
        - is_warning: 是否达到告警阈值
        - warning_threshold: 告警阈值
        - count: API 调用次数
        - success_count: 成功次数
        - fail_count: 失败次数
        - use_for: 用途标识
    """
    usage_info = get_quota_usage_today(daily_quota, use_for)
    usage_rate = usage_info.get("usage_rate", 0.0)
    
    return {
        "used": usage_info.get("used", 0),
        "total": usage_info.get("total", daily_quota),
        "remaining": usage_info.get("remaining", daily_quota),
        "usage_rate": usage_rate,
        "is_exceeded": usage_info.get("remaining", daily_quota) <= 0,
        "is_warning": (usage_rate / 100.0) >= warning_threshold,
        "warning_threshold": warning_threshold * 100,  # 转换为百分比
        "count": usage_info.get("count", 0),
        "success_count": usage_info.get("success_count", 0),
        "fail_count": usage_info.get("fail_count", 0),
        "use_for": use_for,
    }


def check_and_record_quota_warning(
    daily_quota: int = DEFAULT_DAILY_QUOTA,
    warning_threshold: float = QUOTA_WARNING_THRESHOLD,
    check_interval_minutes: int = 60,
    use_for: Optional[str] = None,
) -> bool:
    """
    检查配额使用率并记录告警（CP-y3-04）
    
    如果配额使用率超过阈值，且距离上次告警超过check_interval_minutes分钟，则记录告警。
    避免频繁告警。
    
    Args:
        daily_quota: 每日配额上限
        warning_threshold: 告警阈值（0-1），默认0.8（80%）
        check_interval_minutes: 检查间隔（分钟），默认60分钟
        use_for: API Key 用途标识（"index" 或 "search"），None 表示查询所有或默认
    
    Returns:
        True 如果记录了新告警，False 否则
    """
    usage_info = get_quota_usage_today(daily_quota, use_for)
    usage_rate = usage_info.get("usage_rate", 0.0) / 100.0  # 转换为0-1范围
    
    if usage_rate < warning_threshold:
        return False  # 未达到告警阈值
    
    # 检查最近是否已有告警（避免频繁告警）
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=check_interval_minutes)
        cutoff_time_str = cutoff_time.isoformat()
        
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            # 构建警告类型标识（包含 use_for 信息）
            warning_type = f"usage_rate_high_{use_for}" if use_for else "usage_rate_high"
            cur.execute(
                """
                SELECT COUNT(*) FROM quota_warnings
                WHERE warning_type = ? 
                AND timestamp >= ?
                AND acknowledged = 0
                """,
                (warning_type, cutoff_time_str),
            )
            recent_warnings = cur.fetchone()[0]
            
            if recent_warnings > 0:
                return False  # 最近已有告警，不重复记录
            
            # 记录新告警
            warning_message = (
                f"配额使用率已达到 {usage_rate * 100:.2f}%，"
                f"已使用 {usage_info.get('used', 0)}/{daily_quota}，"
                f"剩余 {usage_info.get('remaining', 0)}"
            )
            
            timestamp = datetime.now(timezone.utc).isoformat()
            warning_type = f"usage_rate_high_{use_for}" if use_for else "usage_rate_high"
            cur.execute(
                """
                INSERT INTO quota_warnings 
                (warning_type, usage_rate, used_quota, total_quota, message, timestamp, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    warning_type,
                    usage_rate * 100,
                    usage_info.get("used", 0),
                    daily_quota,
                    warning_message,
                    timestamp,
                ),
            )
            conn.commit()
            
            logger.warning(f"配额告警: {warning_message}")
            return True
    except Exception as e:
        logger.warning(f"检查配额告警失败: {e}")
        return False


def get_quota_warnings(
    days: int = 7,
    unacknowledged_only: bool = False,
) -> List[Dict]:
    """
    获取配额告警记录
    
    Args:
        days: 查询最近N天的告警
        unacknowledged_only: 是否只返回未确认的告警
    
    Returns:
        告警记录列表
    """
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(day=start_date.day - days + 1)
    start_date_str = start_date.isoformat()
    
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            
            if unacknowledged_only:
                cur.execute(
                    """
                    SELECT id, warning_type, usage_rate, used_quota, total_quota, 
                           message, timestamp, acknowledged
                    FROM quota_warnings
                    WHERE timestamp >= ? AND acknowledged = 0
                    ORDER BY timestamp DESC
                    """,
                    (start_date_str,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, warning_type, usage_rate, used_quota, total_quota, 
                           message, timestamp, acknowledged
                    FROM quota_warnings
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (start_date_str,),
                )
            
            warnings = []
            for row in cur.fetchall():
                warnings.append({
                    "id": row[0],
                    "warning_type": row[1],
                    "usage_rate": row[2],
                    "used_quota": row[3],
                    "total_quota": row[4],
                    "message": row[5],
                    "timestamp": row[6],
                    "acknowledged": bool(row[7]),
                })
            
            return warnings
    except Exception as e:
        logger.warning(f"查询配额告警失败: {e}")
        return []


def acknowledge_quota_warning(warning_id: int) -> bool:
    """
    确认（标记为已处理）配额告警
    
    Args:
        warning_id: 告警ID
    
    Returns:
        True 如果成功，False 否则
    """
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE quota_warnings SET acknowledged = 1 WHERE id = ?",
                (warning_id,),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.warning(f"确认配额告警失败: {e}")
        return False


def get_quota_statistics(days: int = 7, daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> Dict:
    """
    获取配额使用统计信息（CP-y3-10）
    
    Args:
        days: 统计最近N天的数据
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示统计所有
    
    Returns:
        包含以下字段的字典：
        - total_used: 总使用配额
        - total_quota: 总配额（days * daily_quota）
        - average_daily_usage: 平均每日使用量
        - average_daily_rate: 平均每日使用率
        - peak_usage_day: 使用量最高的一天
        - peak_usage_rate: 最高使用率
        - by_endpoint: 按端点统计
        - by_day: 按天统计
        - use_for: 用途标识
    """
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(day=start_date.day - days + 1)
    start_date_str = start_date.isoformat()
    
    try:
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            
            # 总体统计
            if use_for is not None:
                cur.execute(
                    """
                    SELECT 
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count,
                        DATE(timestamp) as usage_date
                    FROM quota_usage
                    WHERE timestamp >= ? AND use_for = ?
                    GROUP BY DATE(timestamp)
                    ORDER BY usage_date DESC
                    """,
                    (start_date_str, use_for),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        COALESCE(SUM(cost), 0) as total_cost,
                        COUNT(*) as total_count,
                        DATE(timestamp) as usage_date
                    FROM quota_usage
                    WHERE timestamp >= ? AND (use_for IS NULL OR use_for = '')
                    GROUP BY DATE(timestamp)
                    ORDER BY usage_date DESC
                    """,
                    (start_date_str,),
                )
            rows = cur.fetchall()
            
            total_used = 0
            by_day = {}
            peak_usage = 0
            peak_usage_day = None
            
            for cost, count, usage_date in rows:
                total_used += int(cost)
                daily_rate = (int(cost) / daily_quota * 100) if daily_quota > 0 else 0
                by_day[usage_date] = {
                    "used": int(cost),
                    "quota": daily_quota,
                    "usage_rate": round(daily_rate, 2),
                    "count": int(count),
                }
                
                if int(cost) > peak_usage:
                    peak_usage = int(cost)
                    peak_usage_day = usage_date
            
            # 按端点统计
            by_endpoint = get_quota_usage_by_endpoint(days=days, use_for=use_for)
            
            total_quota = days * daily_quota
            average_daily_usage = total_used / days if days > 0 else 0
            average_daily_rate = (average_daily_usage / daily_quota * 100) if daily_quota > 0 else 0
            peak_usage_rate = (peak_usage / daily_quota * 100) if daily_quota > 0 else 0
            
            return {
                "total_used": total_used,
                "total_quota": total_quota,
                "average_daily_usage": round(average_daily_usage, 2),
                "average_daily_rate": round(average_daily_rate, 2),
                "peak_usage_day": peak_usage_day,
                "peak_usage": peak_usage,
                "peak_usage_rate": round(peak_usage_rate, 2),
                "by_endpoint": by_endpoint,
                "by_day": by_day,
                "days": days,
                "use_for": use_for,
            }
    except Exception as e:
        logger.warning(f"查询配额统计失败: {e}")
        return {
            "total_used": 0,
            "total_quota": days * daily_quota,
            "average_daily_usage": 0,
            "average_daily_rate": 0,
            "peak_usage_day": None,
            "peak_usage": 0,
            "peak_usage_rate": 0,
            "by_endpoint": {},
            "by_day": {},
            "days": days,
        }


def predict_quota_exhaustion(lookback_hours: int = 1, daily_quota: int = DEFAULT_DAILY_QUOTA, use_for: Optional[str] = None) -> Dict:
    """
    预测配额耗尽时间（CP-y3-11：配额预测）
    
    Args:
        lookback_hours: 用于计算使用速率的回看小时数（1-24）
        daily_quota: 每日配额上限
        use_for: API Key 用途标识（"index" 或 "search"），None 表示统计所有
    
    Returns:
        包含以下字段的字典：
        - current_usage_rate_per_hour: 当前使用速率（单位/小时）
        - remaining_quota: 剩余配额
        - predicted_exhaustion_time: 预测耗尽时间（ISO格式）
        - predicted_exhaustion_hours: 预测耗尽时间（小时数）
        - confidence: 预测置信度说明
        - alert_level: 告警级别（none/low/medium/high/critical）
        - alert_message: 告警消息
    """
    try:
        # 获取当前配额使用情况
        usage = get_quota_usage_today(daily_quota=daily_quota, use_for=use_for)
        used_quota = usage.get("used_quota", 0)
        remaining_quota = daily_quota - used_quota
        
        # 计算最近 lookback_hours 小时的使用速率
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=lookback_hours)
        start_time_str = start_time.isoformat()
        
        with get_quota_db_connection() as conn:
            cur = conn.cursor()
            
            if use_for is not None:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(cost), 0) as total_cost
                    FROM quota_usage
                    WHERE timestamp >= ? AND use_for = ?
                    """,
                    (start_time_str, use_for),
                )
            else:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(cost), 0) as total_cost
                    FROM quota_usage
                    WHERE timestamp >= ? AND (use_for IS NULL OR use_for = '')
                    """,
                    (start_time_str,),
                )
            row = cur.fetchone()
            recent_usage = int(row[0]) if row else 0
        
        # 计算使用速率（单位/小时）
        if lookback_hours > 0:
            usage_rate_per_hour = recent_usage / lookback_hours
        else:
            usage_rate_per_hour = 0
        
        # 预测耗尽时间
        if usage_rate_per_hour > 0:
            predicted_hours = remaining_quota / usage_rate_per_hour
            predicted_time = now + timedelta(hours=predicted_hours)
            predicted_time_str = predicted_time.isoformat()
        else:
            predicted_hours = float('inf')
            predicted_time_str = None
        
        # 确定告警级别
        usage_rate = (used_quota / daily_quota * 100) if daily_quota > 0 else 0
        
        if usage_rate >= 95:
            alert_level = "critical"
            alert_message = f"配额使用率已达 {usage_rate:.1f}%，即将耗尽！"
        elif usage_rate >= 80:
            alert_level = "high"
            alert_message = f"配额使用率已达 {usage_rate:.1f}%，请谨慎使用"
        elif usage_rate >= 60:
            alert_level = "medium"
            alert_message = f"配额使用率为 {usage_rate:.1f}%"
        elif usage_rate >= 40:
            alert_level = "low"
            alert_message = f"配额使用率为 {usage_rate:.1f}%"
        else:
            alert_level = "none"
            alert_message = "配额使用正常"
        
        # 置信度说明
        if lookback_hours >= 6:
            confidence = "高（基于6小时以上数据）"
        elif lookback_hours >= 3:
            confidence = "中（基于3-6小时数据）"
        else:
            confidence = "低（基于少于3小时数据，可能不够准确）"
        
        return {
            "current_usage_rate_per_hour": round(usage_rate_per_hour, 2),
            "remaining_quota": remaining_quota,
            "predicted_exhaustion_time": predicted_time_str,
            "predicted_exhaustion_hours": round(predicted_hours, 2) if predicted_hours != float('inf') else None,
            "confidence": confidence,
            "alert_level": alert_level,
            "alert_message": alert_message,
            "lookback_hours": lookback_hours,
            "used_quota": used_quota,
            "daily_quota": daily_quota,
            "usage_rate": round(usage_rate, 2),
        }
    except Exception as e:
        logger.warning(f"预测配额耗尽时间失败: {e}")
        return {
            "current_usage_rate_per_hour": 0,
            "remaining_quota": daily_quota,
            "predicted_exhaustion_time": None,
            "predicted_exhaustion_hours": None,
            "confidence": "无法计算",
            "alert_level": "none",
            "alert_message": f"预测失败: {str(e)}",
            "lookback_hours": lookback_hours,
            "used_quota": 0,
            "daily_quota": daily_quota,
            "usage_rate": 0,
        }
