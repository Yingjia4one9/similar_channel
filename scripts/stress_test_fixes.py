"""
压力测试脚本：验证修复的线程安全和性能问题
测试以下修复：
1. SQLite线程安全和连接池
2. 缓存失效竞态条件
3. 结果缓存LRU排序
4. 配额跟踪异步检查
"""
import asyncio
import concurrent.futures
import json
import sys
import time
from pathlib import Path

# 强制刷新输出，避免缓冲导致看不到实时日志
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.database import get_db_connection, ensure_schema
from infrastructure.cache import invalidate_channel_videos_cache, _channel_videos_cache
from infrastructure.result_cache import store_result, get_result, _result_cache, _cache_lock
from infrastructure.quota_tracker import record_quota_usage, _quota_record_count


def test_database_connection_pool(num_threads=10, operations_per_thread=5):
    """测试数据库连接池的线程安全性（减少并发数以避免阻塞）"""
    print(f"\n=== 测试1: 数据库连接池线程安全 ===")
    print(f"线程数: {num_threads}, 每线程操作数: {operations_per_thread}")
    
    ensure_schema()
    
    # 预热连接池
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        print(f"预热连接池失败: {e}")
    
    def db_operation(thread_id, op_id):
        """执行数据库操作"""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                # 执行简单的查询
                cur.execute("SELECT COUNT(*) FROM channels")
                count = cur.fetchone()[0]
                # 记录日志
                log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "timestamp": int(time.time() * 1000),
                            "location": "stress_test:db_operation",
                            "message": "数据库操作完成",
                            "data": {"thread_id": thread_id, "op_id": op_id, "count": count},
                            "sessionId": "stress-test",
                            "runId": "stress-test-1",
                            "hypothesisId": "A"
                        }, ensure_ascii=False) + "\n")
                except: pass
                return True
        except Exception as e:
            print(f"线程 {thread_id} 操作 {op_id} 失败: {e}")
            return False
    
    start_time = time.time()
    success_count = 0
    total_ops = num_threads * operations_per_thread
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            for op_id in range(operations_per_thread):
                future = executor.submit(db_operation, thread_id, op_id)
                futures.append(future)
        
        # 添加超时，避免无限等待
        for future in concurrent.futures.as_completed(futures, timeout=120):
            try:
                if future.result(timeout=10):
                    success_count += 1
            except concurrent.futures.TimeoutError:
                print(f"警告: 操作超时")
                success_count += 0
            except Exception as e:
                print(f"警告: 操作异常: {e}")
                success_count += 0
    
    elapsed = time.time() - start_time
    print(f"完成: {success_count}/{total_ops} 操作成功")
    print(f"耗时: {elapsed:.2f}秒")
    print(f"吞吐量: {total_ops/elapsed:.2f} 操作/秒")
    return success_count == total_ops


def test_cache_race_condition(num_threads=10, operations_per_thread=5):
    """测试缓存失效的竞态条件修复"""
    print(f"\n=== 测试2: 缓存失效竞态条件 ===")
    print(f"线程数: {num_threads}, 每线程操作数: {operations_per_thread}")
    
    # 先填充一些缓存
    for i in range(20):
        _channel_videos_cache.set(f"channel_videos:test_channel_{i}:10", {"data": f"test_{i}"})
    
    def cache_operation(thread_id, op_id):
        """执行缓存操作"""
        try:
            channel_id = f"test_channel_{thread_id % 5}"
            # 失效缓存
            invalidate_channel_videos_cache(channel_id)
            # 记录日志
            log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "location": "stress_test:cache_operation",
                        "message": "缓存失效操作完成",
                        "data": {"thread_id": thread_id, "op_id": op_id, "channel_id": channel_id},
                        "sessionId": "stress-test",
                        "runId": "stress-test-2",
                        "hypothesisId": "B"
                    }, ensure_ascii=False) + "\n")
            except: pass
            return True
        except Exception as e:
            print(f"线程 {thread_id} 操作 {op_id} 失败: {e}")
            return False
    
    start_time = time.time()
    success_count = 0
    total_ops = num_threads * operations_per_thread
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            for op_id in range(operations_per_thread):
                future = executor.submit(cache_operation, thread_id, op_id)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
    
    elapsed = time.time() - start_time
    print(f"完成: {success_count}/{total_ops} 操作成功")
    print(f"耗时: {elapsed:.2f}秒")
    return success_count == total_ops


def test_result_cache_lru(num_threads=15, operations_per_thread=10):
    """测试结果缓存LRU排序的线程安全"""
    print(f"\n=== 测试3: 结果缓存LRU排序 ===")
    print(f"线程数: {num_threads}, 每线程操作数: {operations_per_thread}")
    
    def cache_operation(thread_id, op_id):
        """执行缓存操作"""
        try:
            result_id = f"result_{thread_id}_{op_id}"
            result_data = {"data": f"test_{thread_id}_{op_id}"}
            # 存储结果
            store_result(result_id, result_data)
            # 读取结果
            retrieved = get_result(result_id)
            # 记录日志
            log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "location": "stress_test:cache_operation",
                        "message": "结果缓存操作完成",
                        "data": {"thread_id": thread_id, "op_id": op_id, "result_id": result_id, "retrieved": retrieved is not None},
                        "sessionId": "stress-test",
                        "runId": "stress-test-3",
                        "hypothesisId": "C"
                    }, ensure_ascii=False) + "\n")
            except: pass
            return retrieved is not None
        except Exception as e:
            print(f"线程 {thread_id} 操作 {op_id} 失败: {e}")
            return False
    
    start_time = time.time()
    success_count = 0
    total_ops = num_threads * operations_per_thread
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            for op_id in range(operations_per_thread):
                future = executor.submit(cache_operation, thread_id, op_id)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
    
    elapsed = time.time() - start_time
    print(f"完成: {success_count}/{total_ops} 操作成功")
    print(f"耗时: {elapsed:.2f}秒")
    return success_count == total_ops


def test_quota_tracker_race_condition(num_threads=10, operations_per_thread=3):
    """测试配额跟踪的异步检查竞态条件修复"""
    print(f"\n=== 测试4: 配额跟踪异步检查 ===")
    print(f"线程数: {num_threads}, 每线程操作数: {operations_per_thread}")
    
    def quota_operation(thread_id, op_id):
        """执行配额操作"""
        try:
            # 记录配额使用
            record_quota_usage(
                endpoint="test",
                method="list",
                cost=1,
                params={"thread_id": thread_id, "op_id": op_id},
                success=True,
                use_for="test"
            )
            # 记录日志
            log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "location": "stress_test:quota_operation",
                        "message": "配额记录操作完成",
                        "data": {"thread_id": thread_id, "op_id": op_id},
                        "sessionId": "stress-test",
                        "runId": "stress-test-4",
                        "hypothesisId": "D"
                    }, ensure_ascii=False) + "\n")
            except: pass
            return True
        except Exception as e:
            print(f"线程 {thread_id} 操作 {op_id} 失败: {e}")
            return False
    
    start_time = time.time()
    success_count = 0
    total_ops = num_threads * operations_per_thread
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            for op_id in range(operations_per_thread):
                future = executor.submit(quota_operation, thread_id, op_id)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
    
    elapsed = time.time() - start_time
    print(f"完成: {success_count}/{total_ops} 操作成功")
    print(f"耗时: {elapsed:.2f}秒")
    return success_count == total_ops


def main():
    """运行所有压力测试"""
    print("=" * 60)
    print("开始压力测试：验证修复的线程安全和性能问题")
    print("=" * 60)
    
    # 清空日志文件
    log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("")  # 清空文件
        print(f"已清空日志文件: {log_path}")
    except Exception as e:
        print(f"清空日志文件失败: {e}")
    
    results = {}
    
    # 测试1: 数据库连接池
    try:
        results['database_pool'] = test_database_connection_pool(num_threads=20, operations_per_thread=10)
    except Exception as e:
        print(f"测试1失败: {e}")
        results['database_pool'] = False
    
    # 测试2: 缓存失效竞态条件
    try:
        results['cache_race'] = test_cache_race_condition(num_threads=10, operations_per_thread=5)
    except Exception as e:
        print(f"测试2失败: {e}")
        results['cache_race'] = False
    
    # 测试3: 结果缓存LRU
    try:
        results['result_cache_lru'] = test_result_cache_lru(num_threads=15, operations_per_thread=10)
    except Exception as e:
        print(f"测试3失败: {e}")
        results['result_cache_lru'] = False
    
    # 测试4: 配额跟踪异步检查
    try:
        results['quota_tracker'] = test_quota_tracker_race_condition(num_threads=20, operations_per_thread=5)
    except Exception as e:
        print(f"测试4失败: {e}")
        results['quota_tracker'] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Fixes verified successfully.")
    else:
        print("[FAILED] Some tests failed, please check the log file.")
    print("=" * 60)
    print(f"\n详细日志已保存到: {log_path}")
    
    return all_passed


if __name__ == "__main__":
    main()

