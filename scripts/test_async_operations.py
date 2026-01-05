"""
异步操作测试脚本：验证项目中的异步功能是否正常工作

测试内容：
1. asyncio.gather 并发执行
2. asyncio.Queue 用于进度更新
3. asyncio.create_task 创建后台任务
4. encode_async 异步编码函数
5. 异步数据库操作（如果可用）
6. 异步任务超时和异常处理
"""
import asyncio
import sys
import time
import io
from pathlib import Path
from typing import List, Dict, Any

# 设置 UTF-8 编码以支持中文字符
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 强制刷新输出，避免缓冲导致看不到实时日志
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_asyncio_gather():
    """测试 asyncio.gather 并发执行"""
    print("\n=== 测试1: asyncio.gather 并发执行 ===")
    
    async def task(name: str, delay: float) -> str:
        """模拟一个异步任务"""
        await asyncio.sleep(delay)
        return f"任务 {name} 完成"
    
    start_time = time.time()
    
    # 创建多个并发任务
    tasks = [
        task("A", 0.1),
        task("B", 0.2),
        task("C", 0.15),
        task("D", 0.05),
    ]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    print(f"所有任务完成: {results}")
    print(f"总耗时: {elapsed:.3f}秒 (如果是并发，应该约等于最长的任务时间 ~0.2秒)")
    
    # 验证并发执行（总时间应该接近最长任务的时间，而不是所有任务时间的总和）
    assert elapsed < 0.25, f"并发执行失败，耗时 {elapsed} 秒"
    assert len(results) == 4, "应该返回4个结果"
    
    print("[PASS] asyncio.gather 测试通过")
    return True


async def test_asyncio_gather_with_exceptions():
    """测试 asyncio.gather 的异常处理"""
    print("\n=== 测试2: asyncio.gather 异常处理 ===")
    
    async def success_task(name: str) -> str:
        await asyncio.sleep(0.05)
        return f"成功: {name}"
    
    async def fail_task(name: str) -> str:
        await asyncio.sleep(0.05)
        raise ValueError(f"任务 {name} 失败")
    
    # 使用 return_exceptions=True 来捕获异常
    results = await asyncio.gather(
        success_task("A"),
        fail_task("B"),
        success_task("C"),
        return_exceptions=True
    )
    
    print(f"结果: {results}")
    
    # 验证结果
    assert results[0] == "成功: A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "成功: C"
    
    print("[PASS] asyncio.gather 异常处理测试通过")
    return True


async def test_asyncio_queue():
    """测试 asyncio.Queue 用于进度更新"""
    print("\n=== 测试3: asyncio.Queue 进度更新 ===")
    
    progress_queue = asyncio.Queue()
    progress_updates = []
    
    async def producer():
        """生产者：模拟任务进度"""
        for i in range(5):
            await asyncio.sleep(0.1)
            progress = (i + 1) * 20
            message = f"处理进度 {progress}%"
            await progress_queue.put((progress, message))
            print(f"  生产: {progress}% - {message}")
        await progress_queue.put(None)  # 发送结束信号
    
    async def consumer():
        """消费者：接收进度更新"""
        while True:
            item = await progress_queue.get()
            if item is None:
                break
            progress, message = item
            progress_updates.append((progress, message))
            print(f"  消费: {progress}% - {message}")
            progress_queue.task_done()
    
    # 并发运行生产者和消费者
    await asyncio.gather(producer(), consumer())
    
    # 验证进度更新
    assert len(progress_updates) == 5, f"应该有5个进度更新，实际收到 {len(progress_updates)}"
    assert progress_updates[-1][0] == 100, "最后一个进度应该是100%"
    
    print("[PASS] asyncio.Queue 测试通过")
    return True


async def test_asyncio_create_task():
    """测试 asyncio.create_task 创建后台任务"""
    print("\n=== 测试4: asyncio.create_task 后台任务 ===")
    
    task_results = []
    
    async def background_task(task_id: int, delay: float):
        """后台任务"""
        await asyncio.sleep(delay)
        result = f"后台任务 {task_id} 完成"
        task_results.append(result)
        return result
    
    # 创建后台任务
    task1 = asyncio.create_task(background_task(1, 0.2))
    task2 = asyncio.create_task(background_task(2, 0.1))
    
    # 验证任务已创建但未完成
    assert not task1.done()
    assert not task2.done()
    
    # 等待任务完成
    results = await asyncio.gather(task1, task2)
    
    # 验证任务已完成
    assert task1.done()
    assert task2.done()
    assert len(results) == 2
    assert len(task_results) == 2
    
    print(f"任务结果: {results}")
    print("[PASS] asyncio.create_task 测试通过")
    return True


async def test_encode_async():
    """测试 encode_async 异步编码函数"""
    print("\n=== 测试5: encode_async 异步编码 ===")
    
    try:
        from core.embedding import encode_async
        
        # 测试单个文本编码
        texts = ["这是一个测试文本"]
        print(f"编码文本: {texts}")
        
        start_time = time.time()
        vectors = await encode_async(texts)
        elapsed = time.time() - start_time
        
        print(f"编码完成，耗时: {elapsed:.3f}秒")
        print(f"向量形状: {vectors.shape}")
        print(f"向量维度: {vectors.shape[1] if len(vectors.shape) > 1 else len(vectors)}")
        
        # 验证结果
        assert vectors is not None, "向量不应为 None"
        assert len(vectors) == 1, "应该返回1个向量"
        assert vectors.shape[0] == 1, "向量数组应该有1行"
        
        # 测试批量编码
        batch_texts = ["文本1", "文本2", "文本3"]
        print(f"\n批量编码文本: {batch_texts}")
        
        start_time = time.time()
        batch_vectors = await encode_async(batch_texts)
        elapsed = time.time() - start_time
        
        print(f"批量编码完成，耗时: {elapsed:.3f}秒")
        print(f"向量形状: {batch_vectors.shape}")
        
        assert batch_vectors.shape[0] == 3, "应该有3个向量"
        
        print("[PASS] encode_async 测试通过")
        return True
        
    except ImportError as e:
        print(f"[WARN] 无法导入 encode_async: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] encode_async 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_db_operations():
    """测试异步数据库操作"""
    print("\n=== 测试6: 异步数据库操作 ===")
    
    try:
        from infrastructure.database import get_async_db_connection
        
        # 测试异步数据库连接
        async with get_async_db_connection() as db:
            # 执行简单查询
            async with db.execute("SELECT 1 as test") as cursor:
                rows = await cursor.fetchall()
                assert len(rows) == 1
                assert rows[0]['test'] == 1
                print("  异步查询成功: SELECT 1")
            
            # 测试表是否存在
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='channels'") as cursor:
                rows = await cursor.fetchall()
                if rows:
                    print("  channels 表存在")
                    # 查询表记录数
                    async with db.execute("SELECT COUNT(*) as count FROM channels") as cursor:
                        count_row = await cursor.fetchone()
                        if count_row:
                            count = count_row['count']
                            print(f"  channels 表中有 {count} 条记录")
                else:
                    print("  channels 表不存在（这是正常的，如果数据库未初始化）")
        
        print("[PASS] 异步数据库操作测试通过")
        return True
        
    except ImportError as e:
        print(f"[WARN] 无法导入异步数据库函数: {e}")
        return False
    except Exception as e:
        print(f"[WARN] 异步数据库操作测试失败（可能是数据库未初始化）: {e}")
        return False


async def test_async_timeout():
    """测试异步任务超时处理"""
    print("\n=== 测试7: 异步任务超时处理 ===")
    
    async def slow_task():
        """慢速任务"""
        await asyncio.sleep(0.3)
        return "完成"
    
    # 测试超时
    try:
        result = await asyncio.wait_for(slow_task(), timeout=0.1)
        print(f"[FAIL] 不应该到达这里: {result}")
        return False
    except asyncio.TimeoutError:
        print("  正确捕获了超时异常")
    
    # 测试正常完成
    try:
        result = await asyncio.wait_for(slow_task(), timeout=0.5)
        assert result == "完成"
        print("  任务正常完成")
    except asyncio.TimeoutError:
        print("[FAIL] 不应该超时")
        return False
    
    print("[PASS] 异步任务超时处理测试通过")
    return True


async def test_concurrent_encode():
    """测试并发编码任务"""
    print("\n=== 测试8: 并发编码任务 ===")
    
    try:
        from core.embedding import encode_async
        
        # 创建多个编码任务
        text_batches = [
            ["批次1-文本1", "批次1-文本2"],
            ["批次2-文本1", "批次2-文本2"],
            ["批次3-文本1", "批次3-文本2"],
        ]
        
        # 串行执行（对比基准）
        start_serial = time.time()
        serial_results = []
        for batch in text_batches:
            vectors = await encode_async(batch)
            serial_results.append(vectors)
        serial_time = time.time() - start_serial
        print(f"串行执行耗时: {serial_time:.3f}秒")
        
        # 并发执行
        start_parallel = time.time()
        parallel_results = await asyncio.gather(*[encode_async(batch) for batch in text_batches])
        parallel_time = time.time() - start_parallel
        print(f"并发执行耗时: {parallel_time:.3f}秒")
        
        # 验证结果
        assert len(parallel_results) == 3
        assert len(serial_results) == 3
        
        # 并发执行应该更快（虽然编码本身在线程池中，但仍有优化空间）
        print(f"性能提升: {(serial_time / parallel_time - 1) * 100:.1f}%")
        
        print("[PASS] 并发编码任务测试通过")
        return True
        
    except ImportError as e:
        print(f"[WARN] 无法导入 encode_async: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] 并发编码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_progress_callback_pattern():
    """测试进度回调模式（模拟主应用中的用法）"""
    print("\n=== 测试9: 进度回调模式 ===")
    
    progress_queue = asyncio.Queue()
    received_progress = []
    
    def progress_callback(progress: float, message: str):
        """进度回调函数"""
        try:
            progress_queue.put_nowait((progress, message))
        except Exception:
            pass  # 忽略队列满的错误
    
    async def simulated_work(progress_callback_func):
        """模拟工作流程"""
        for i in range(5):
            await asyncio.sleep(0.1)
            progress = (i + 1) * 20
            message = f"步骤 {i + 1}/5"
            progress_callback_func(progress, message)
    
    async def progress_listener():
        """进度监听器"""
        while True:
            try:
                progress, message = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                received_progress.append((progress, message))
                print(f"  收到进度: {progress}% - {message}")
            except asyncio.TimeoutError:
                break
    
    # 并发运行工作和监听器
    await asyncio.gather(
        simulated_work(progress_callback),
        progress_listener()
    )
    
    # 验证进度更新
    assert len(received_progress) >= 4, f"应该至少收到4个进度更新，实际收到 {len(received_progress)}"
    
    print("[PASS] 进度回调模式测试通过")
    return True


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("异步操作功能测试")
    print("=" * 60)
    
    tests = [
        ("asyncio.gather 并发执行", test_asyncio_gather),
        ("asyncio.gather 异常处理", test_asyncio_gather_with_exceptions),
        ("asyncio.Queue 进度更新", test_asyncio_queue),
        ("asyncio.create_task 后台任务", test_asyncio_create_task),
        ("encode_async 异步编码", test_encode_async),
        ("异步数据库操作", test_async_db_operations),
        ("异步任务超时处理", test_async_timeout),
        ("并发编码任务", test_concurrent_encode),
        ("进度回调模式", test_progress_callback_pattern),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] 测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS] 通过" if result else "[FAIL] 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有异步操作测试通过！")
        return 0
    else:
        print(f"\n[WARN] 有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

