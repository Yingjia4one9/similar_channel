"""
清理旧配额记录脚本
清理添加 use_for 字段之前的旧记录（use_for IS NULL 的记录）
"""
import sys
from infrastructure.quota_tracker import clean_legacy_quota_records
from infrastructure.logger import get_logger

logger = get_logger()


def main():
    """主函数"""
    print("=" * 60)
    print("清理旧配额记录工具")
    print("=" * 60)
    print()
    print("此工具将清理添加 use_for 字段之前的旧记录（use_for IS NULL）")
    print("这些旧记录会影响默认用途的配额统计。")
    print()
    
    # 先试运行，查看有多少记录
    print("正在检查旧记录...")
    dry_run_result = clean_legacy_quota_records(dry_run=True)
    
    if dry_run_result.get("total_count", 0) == 0:
        print("✓ 没有发现旧记录，无需清理。")
        return 0
    
    total_count = dry_run_result.get("total_count", 0)
    print(f"发现 {total_count} 条旧记录待清理。")
    print()
    
    # 询问用户确认
    response = input("是否继续清理？(y/N): ").strip().lower()
    if response not in ('y', 'yes'):
        print("已取消清理操作。")
        return 0
    
    print()
    print("正在清理旧记录...")
    result = clean_legacy_quota_records(dry_run=False)
    
    deleted_count = result.get("deleted_count", 0)
    if deleted_count > 0:
        print(f"✓ 成功清理了 {deleted_count} 条旧记录。")
        print()
        print("现在默认用途（use_for IS NULL）的统计将只包含")
        print("真正使用默认 API Key 的记录。")
    else:
        print("⚠ 清理失败或没有记录被删除。")
        if "error" in result:
            print(f"错误信息: {result['error']}")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n操作已取消。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"清理旧记录时出错: {e}", exc_info=True)
        print(f"\n错误: {e}")
        sys.exit(1)

