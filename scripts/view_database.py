"""查看数据库内容的工具脚本"""
import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 数据库路径
DB_PATH = Path(__file__).parent.parent / "data" / "channel_index.db"

def view_database():
    """查看数据库内容"""
    if not DB_PATH.exists():
        print(f"数据库文件不存在: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print("=" * 80)
    print("数据库内容查看工具")
    print("=" * 80)
    print(f"数据库路径: {DB_PATH}")
    print()
    
    # 1. 查看表结构
    print("【表结构】")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    for table in tables:
        table_name = table[0]
        print(f"\n表: {table_name}")
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()
        print("  列:")
        for col in columns:
            print(f"    - {col[1]} ({col[2]})")
    
    # 2. 统计信息
    print("\n" + "=" * 80)
    print("【统计信息】")
    
    # channels 表统计
    cur.execute("SELECT COUNT(*) as count FROM channels")
    channel_count = cur.fetchone()[0]
    print(f"频道总数: {channel_count}")
    
    if channel_count > 0:
        cur.execute("SELECT COUNT(*) as count FROM channels WHERE updated_at IS NOT NULL")
        updated_count = cur.fetchone()[0]
        print(f"已更新频道数: {updated_count}")
        
        cur.execute("SELECT MIN(updated_at) as min_date, MAX(updated_at) as max_date FROM channels WHERE updated_at IS NOT NULL")
        date_range = cur.fetchone()
        if date_range[0]:
            print(f"最早更新: {date_range[0]}")
            print(f"最新更新: {date_range[1]}")
    
    # channel_embeddings 表统计
    cur.execute("SELECT COUNT(*) as count FROM channel_embeddings")
    embedding_count = cur.fetchone()[0]
    print(f"向量总数: {embedding_count}")
    
    # 3. 查看部分频道数据
    print("\n" + "=" * 80)
    print("【频道数据示例（前10条）】")
    
    cur.execute("""
        SELECT 
            channel_id, 
            title, 
            subscriber_count, 
            view_count,
            country,
            language,
            updated_at
        FROM channels 
        ORDER BY updated_at DESC 
        LIMIT 10
    """)
    
    channels = cur.fetchall()
    if channels:
        for i, ch in enumerate(channels, 1):
            print(f"\n{i}. 频道ID: {ch['channel_id']}")
            print(f"   标题: {ch['title'] or '(无)'}")
            print(f"   订阅数: {ch['subscriber_count'] or 0:,}")
            print(f"   观看数: {ch['view_count'] or 0:,}")
            print(f"   国家: {ch['country'] or '(无)'}")
            print(f"   语言: {ch['language'] or '(无)'}")
            print(f"   更新时间: {ch['updated_at'] or '(无)'}")
    else:
        print("   (无数据)")
    
    # 4. 查看topics和audience分布
    print("\n" + "=" * 80)
    print("【Topics分布（前10个）】")
    
    cur.execute("""
        SELECT topics, COUNT(*) as count 
        FROM channels 
        WHERE topics IS NOT NULL AND topics != ''
        GROUP BY topics 
        ORDER BY count DESC 
        LIMIT 10
    """)
    
    topics_dist = cur.fetchall()
    if topics_dist:
        for topic_row in topics_dist:
            try:
                topics = json.loads(topic_row['topics'])
                topics_str = ', '.join(topics[:3]) if topics else '(无)'
                if len(topics) > 3:
                    topics_str += f" ... (+{len(topics)-3}个)"
                print(f"  {topics_str}: {topic_row['count']} 个频道")
            except:
                print(f"  (解析失败): {topic_row['count']} 个频道")
    else:
        print("   (无数据)")
    
    # 5. 按订阅数排序的top频道
    print("\n" + "=" * 80)
    print("【订阅数Top 10频道】")
    
    cur.execute("""
        SELECT 
            channel_id,
            title,
            subscriber_count,
            view_count
        FROM channels
        WHERE subscriber_count IS NOT NULL
        ORDER BY subscriber_count DESC
        LIMIT 10
    """)
    
    top_channels = cur.fetchall()
    if top_channels:
        for i, ch in enumerate(top_channels, 1):
            print(f"{i:2d}. {ch['title'] or ch['channel_id']}")
            print(f"     订阅数: {ch['subscriber_count']:,}  |  观看数: {ch['view_count']:,}")
    else:
        print("   (无数据)")
    
    conn.close()
    print("\n" + "=" * 80)
    print("查看完成！")

if __name__ == "__main__":
    view_database()

