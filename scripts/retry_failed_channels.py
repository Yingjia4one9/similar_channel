#!/usr/bin/env python3
"""
重新处理失败的频道数据。

当批量更新数据库失败时，数据会保留在内存中，但脚本结束后会丢失。
此脚本用于重新处理那些可能因为数据库表结构问题而失败的频道。

使用方法：
1. 先运行 fix_database_schema.py 修复数据库表结构
2. 然后重新运行 build_channel_index.py，它会自动处理需要更新的频道
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infrastructure.logger import get_logger
from infrastructure.database import get_db_connection
from scripts.build_channel_index import _channel_needs_update

logger = get_logger()

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "channel_index.db"))


def find_channels_needing_update(max_age_days: int = 60):
    """查找需要更新的频道ID列表"""
    if not os.path.exists(DB_PATH):
        logger.warning("数据库文件不存在，无法查找需要更新的频道")
        return []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # 查找所有频道ID
            cur.execute("SELECT channel_id FROM channels")
            all_channel_ids = [row[0] for row in cur.fetchall()]
            
            # 检查哪些需要更新
            needs_update = []
            for channel_id in all_channel_ids:
                if _channel_needs_update(conn, channel_id, max_age_days):
                    needs_update.append(channel_id)
            
            logger.info(f"找到 {len(needs_update)} 个需要更新的频道（共 {len(all_channel_ids)} 个）")
            return needs_update
            
    except Exception as e:
        logger.error(f"查找需要更新的频道失败: {e}", exc_info=True)
        return []


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("重新处理失败的频道数据")
    logger.info("=" * 60)
    
    # 查找需要更新的频道
    channel_ids = find_channels_needing_update(max_age_days=0)  # 0 表示强制更新所有频道
    
    if not channel_ids:
        logger.info("没有需要更新的频道")
        return
    
    logger.info(f"\n找到 {len(channel_ids)} 个需要重新处理的频道")
    logger.info("请运行以下命令重新处理这些频道：")
    logger.info(f"python scripts/build_channel_index.py")
    logger.info("\n或者，如果您想强制更新所有频道，可以运行：")
    logger.info(f"python scripts/build_channel_index.py --force-update")


if __name__ == "__main__":
    main()

