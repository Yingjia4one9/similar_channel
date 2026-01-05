"""
直接清理 index 的今天配额记录（非交互式）
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from infrastructure.quota_tracker import clear_today_quota_records, get_quota_usage_today, DEFAULT_DAILY_QUOTA

def main():
    print("=" * 60)
    print("清理 index 的今天配额记录")
    print("=" * 60)
    
    # 显示清理前的配额使用情况
    usage_before = get_quota_usage_today(daily_quota=DEFAULT_DAILY_QUOTA, use_for="index")
    print(f"\n清理前 [index]:")
    print(f"  已使用: {usage_before['used']}")
    print(f"  使用率: {usage_before['usage_rate']:.2f}%")
    print(f"  调用次数: {usage_before['count']}")
    
    # 执行清理
    print(f"\n正在清理 [index] 的今天配额记录...")
    result = clear_today_quota_records(use_for="index")
    
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"成功清理了 {result['deleted_count']} 条记录")
    
    # 显示清理后的配额使用情况
    usage_after = get_quota_usage_today(daily_quota=DEFAULT_DAILY_QUOTA, use_for="index")
    print(f"\n清理后 [index]:")
    print(f"  已使用: {usage_after['used']}")
    print(f"  使用率: {usage_after['usage_rate']:.2f}%")
    print(f"  调用次数: {usage_after['count']}")

if __name__ == "__main__":
    main()

