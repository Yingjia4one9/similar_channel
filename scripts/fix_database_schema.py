#!/usr/bin/env python3
"""
修复数据库表结构，添加缺失的字段（如 recent_videos, engagement_rate 等）
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from infrastructure.logger import get_logger

logger = get_logger()

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "channel_index.db"))


def fix_schema():
    """修复数据库表结构，添加缺失的字段"""
    if not os.path.exists(DB_PATH):
        logger.error(f"数据库文件不存在: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # 检查当前字段
        cur.execute("PRAGMA table_info(channels)")
        existing_columns = [row[1] for row in cur.fetchall()]
        logger.info(f"当前 channels 表字段: {existing_columns}")
        
        # 需要添加的字段
        fields_to_add = [
            ("recent_videos", "TEXT"),
            ("engagement_rate", "REAL DEFAULT 0.0 CHECK(engagement_rate >= 0.0)"),
            ("view_rate", "REAL DEFAULT 0.0 CHECK(view_rate >= 0.0)"),
            ("competitor_detection", "TEXT"),
        ]
        
        added_fields = []
        for field_name, field_type in fields_to_add:
            if field_name not in existing_columns:
                try:
                    cur.execute(f"ALTER TABLE channels ADD COLUMN {field_name} {field_type}")
                    added_fields.append(field_name)
                    logger.info(f"[OK] 已添加字段: {field_name}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"添加字段 {field_name} 失败: {e}")
            else:
                logger.debug(f"字段 {field_name} 已存在，跳过")
        
        conn.commit()
        conn.close()
        
        if added_fields:
            logger.info(f"成功添加 {len(added_fields)} 个字段: {added_fields}")
            return True
        else:
            logger.info("所有字段都已存在，无需修复")
            return True
            
    except Exception as e:
        logger.error(f"修复数据库表结构失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("开始修复数据库表结构...")
    success = fix_schema()
    if success:
        logger.info("数据库表结构修复完成！")
        sys.exit(0)
    else:
        logger.error("数据库表结构修复失败！")
        sys.exit(1)

