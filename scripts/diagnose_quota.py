"""
诊断配额统计问题
检查数据库中的配额记录和计算逻辑
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# 添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from infrastructure.quota_tracker import get_quota_usage_today, DEFAULT_DAILY_QUOTA, QUOTA_DB_PATH, get_pacific_time_today_start

def diagnose_quota():
    """诊断配额统计问题"""
    print("=" * 60)
    print("配额统计诊断")
    print("=" * 60)
    
    # 1. 检查当前时间
    now_utc = datetime.now(timezone.utc)
    pt_today_start_utc = get_pacific_time_today_start()
    old_utc_today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    
    print(f"\n当前 UTC 时间: {now_utc.isoformat()}")
    print(f"太平洋时间今天开始 (UTC): {pt_today_start_utc.isoformat()}")
    print(f"旧逻辑 UTC 00:00: {old_utc_today_start.isoformat()}")
    print(f"本地时间: {datetime.now().isoformat()}")
    print(f"\n注意: YouTube 配额在太平洋时间 00:00 刷新")
    print(f"      - 冬令时 (PST UTC-8): PT 00:00 = UTC 08:00")
    print(f"      - 夏令时 (PDT UTC-7): PT 00:00 = UTC 07:00")
    
    # 2. 查询程序计算的配额使用
    print("\n" + "-" * 60)
    print("程序计算的配额使用情况:")
    print("-" * 60)
    
    for use_for in [None, "index", "search"]:
        usage = get_quota_usage_today(daily_quota=DEFAULT_DAILY_QUOTA, use_for=use_for)
        label = use_for if use_for else "default"
        print(f"\n[{label}]")
        print(f"  已使用: {usage['used']}")
        print(f"  总配额: {usage['total']}")
        print(f"  剩余: {usage['remaining']}")
        print(f"  使用率: {usage['usage_rate']:.2f}%")
        print(f"  调用次数: {usage['count']}")
    
    # 3. 直接查询数据库
    print("\n" + "-" * 60)
    print("数据库中的实际记录:")
    print("-" * 60)
    
    if not os.path.exists(QUOTA_DB_PATH):
        print(f"数据库文件不存在: {QUOTA_DB_PATH}")
        return
    
    conn = sqlite3.connect(QUOTA_DB_PATH)
    cur = conn.cursor()
    
    # 检查表结构
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quota_usage'")
    if not cur.fetchone():
        print("quota_usage 表不存在")
        conn.close()
        return
    
    # 查询今天的记录（使用太平洋时间）
    today_start_str = pt_today_start_utc.isoformat()
    print(f"\n查询条件 (太平洋时间今天开始): timestamp >= {today_start_str}")
    
    # 查询所有今天的记录（按 use_for 分组）
    for use_for in [None, "index", "search"]:
        if use_for is not None:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    COALESCE(SUM(cost), 0) as total_cost,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM quota_usage
                WHERE timestamp >= ? AND use_for = ?
                """,
                (today_start_str, use_for)
            )
        else:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    COALESCE(SUM(cost), 0) as total_cost,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM quota_usage
                WHERE timestamp >= ? AND (use_for IS NULL OR use_for = '')
                """,
                (today_start_str,)
            )
        
        row = cur.fetchone()
        label = use_for if use_for else "default"
        print(f"\n[{label}] 今天的记录:")
        if row and row[0] > 0:
            print(f"  记录数: {row[0]}")
            print(f"  总消耗: {row[1]}")
            print(f"  最早记录: {row[2]}")
            print(f"  最晚记录: {row[3]}")
        else:
            print("  无记录")
    
    # 查询昨天的记录（可能被误算到今天）
    from datetime import timedelta
    yesterday_start = pt_today_start_utc - timedelta(days=1)
    yesterday_start_str = yesterday_start.isoformat()
    
    print(f"\n查询昨天的记录: timestamp >= {yesterday_start_str} AND timestamp < {today_start_str}")
    
    for use_for in [None, "index", "search"]:
        if use_for is not None:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    COALESCE(SUM(cost), 0) as total_cost,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM quota_usage
                WHERE timestamp >= ? AND timestamp < ? AND use_for = ?
                """,
                (yesterday_start_str, today_start_str, use_for)
            )
        else:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    COALESCE(SUM(cost), 0) as total_cost,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM quota_usage
                WHERE timestamp >= ? AND timestamp < ? AND (use_for IS NULL OR use_for = '')
                """,
                (yesterday_start_str, today_start_str)
            )
        
        row = cur.fetchone()
        label = use_for if use_for else "default"
        if row and row[0] > 0:
            print(f"\n[{label}] 昨天的记录 (可能被误算):")
            print(f"  记录数: {row[0]}")
            print(f"  总消耗: {row[1]}")
            print(f"  最早记录: {row[2]}")
            print(f"  最晚记录: {row[3]}")
    
    # 查询最近的几条记录（查看时间格式）
    print(f"\n最近的 10 条记录 (所有 use_for):")
    cur.execute(
        """
        SELECT timestamp, endpoint, cost, use_for
        FROM quota_usage
        ORDER BY timestamp DESC
        LIMIT 10
        """
    )
    for row in cur.fetchall():
        ts, endpoint, cost, use_for_val = row
        print(f"  {ts} | {endpoint} | cost={cost} | use_for={use_for_val}")
    
    conn.close()
    
    # 4. 建议
    print("\n" + "=" * 60)
    print("诊断建议:")
    print("=" * 60)
    print("1. 如果 YouTube 显示配额已重置为 0，但程序仍显示高使用率，")
    print("   可能是数据库中的旧记录没有被正确过滤")
    print("2. 检查时间戳格式是否正确（应该是 ISO 格式，带时区信息）")
    print("3. 如果确认 YouTube 配额已重置，可以手动清理数据库中的旧记录")

if __name__ == "__main__":
    diagnose_quota()

