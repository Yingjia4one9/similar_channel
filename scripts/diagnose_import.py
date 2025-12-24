"""诊断导入阻塞问题"""
import sys
import time

print("开始诊断导入过程...")
print("=" * 60)

# 记录每个导入的时间
imports_to_test = [
    ("json", "标准库"),
    ("os", "标准库"),
    ("sqlite3", "标准库"),
    ("sys", "标准库"),
    ("time", "标准库"),
    ("concurrent.futures", "标准库"),
    ("datetime", "标准库"),
    ("threading", "标准库"),
    ("typing", "标准库"),
    ("numpy", "第三方库"),
    ("infrastructure.logger", "项目模块"),
    ("infrastructure.config", "项目模块"),
    ("infrastructure.database", "项目模块"),
    ("infrastructure.quota_tracker", "项目模块"),
    ("core.embedding", "项目模块"),
    ("core.youtube_api", "项目模块"),
    ("core.channel_info", "项目模块"),
]

results = []

for module_name, category in imports_to_test:
    start = time.time()
    try:
        if "." in module_name:
            # 处理子模块导入
            parts = module_name.split(".")
            mod = __import__(module_name, fromlist=[parts[-1]])
        else:
            mod = __import__(module_name)
        elapsed = time.time() - start
        status = "OK"
        if elapsed > 0.1:
            status = "SLOW"
        if elapsed > 1.0:
            status = "VERY_SLOW"
        results.append((module_name, category, elapsed, status, None))
        print(f"{status:10s} {module_name:30s} ({category:10s}) - {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start
        results.append((module_name, category, elapsed, "FAILED", str(e)))
        print(f"FAILED {module_name:30s} ({category:10s}) - {elapsed:.3f}s - Error: {e}")

print("=" * 60)
print("\n慢速导入汇总 (>0.1秒):")
slow_imports = [r for r in results if r[2] > 0.1]
if slow_imports:
    for name, cat, elapsed, status, error in sorted(slow_imports, key=lambda x: x[2], reverse=True):
        print(f"  {name:30s} - {elapsed:.3f}秒")
else:
    print("  无慢速导入")

print("\n测试完整导入 build_channel_index...")
start = time.time()
try:
    sys.path.insert(0, '.')
    from scripts.build_channel_index import build_index
    elapsed = time.time() - start
    print(f"OK Complete import successful - {elapsed:.3f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"FAILED Complete import failed - {elapsed:.3f}s")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

