"""
重置今天的配额统计
当 YouTube 配额已重置但程序仍显示高使用率时，可以使用此脚本清理今天的记录
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
    print("重置今天的配额统计")
    print("=" * 60)
    
    # 显示当前配额使用情况
    print("\n当前配额使用情况:")
    print("-" * 60)
    
    for use_for in [None, "index", "search"]:
        usage = get_quota_usage_today(daily_quota=DEFAULT_DAILY_QUOTA, use_for=use_for)
        label = use_for if use_for else "default"
        print(f"\n[{label}]")
        print(f"  已使用: {usage['used']}")
        print(f"  使用率: {usage['usage_rate']:.2f}%")
        print(f"  调用次数: {usage['count']}")
    
    # 询问用户要清理哪些记录
    print("\n" + "=" * 60)
    print("请选择要清理的记录:")
    print("1. 清理所有 (default, index, search)")
    print("2. 只清理 index")
    print("3. 只清理 search")
    print("4. 取消")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        use_for = None
        label = "所有"
    elif choice == "2":
        use_for = "index"
        label = "index"
    elif choice == "3":
        use_for = "search"
        label = "search"
    else:
        print("已取消")
        return
    
    # 确认
    confirm = input(f"\n确认要清理 [{label}] 的今天配额记录吗？(yes/no): ").strip().lower()
    if confirm != "yes":
        print("已取消")
        return
    
    # 执行清理
    print(f"\n正在清理 [{label}] 的今天配额记录...")
    result = clear_today_quota_records(use_for=use_for)
    
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"成功清理了 {result['deleted_count']} 条记录")
    
    # 显示清理后的配额使用情况
    print("\n清理后的配额使用情况:")
    print("-" * 60)
    
    for check_for in [None, "index", "search"]:
        usage = get_quota_usage_today(daily_quota=DEFAULT_DAILY_QUOTA, use_for=check_for)
        label = check_for if check_for else "default"
        print(f"\n[{label}]")
        print(f"  已使用: {usage['used']}")
        print(f"  使用率: {usage['usage_rate']:.2f}%")
        print(f"  调用次数: {usage['count']}")

if __name__ == "__main__":
    main()

