"""测试日志系统是否正常工作"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试日志写入
log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
log_dir = os.path.dirname(log_path)

print(f"日志目录: {log_dir}")
print(f"日志文件: {log_path}")
print(f"目录存在: {os.path.exists(log_dir)}")

# 确保目录存在
if not os.path.exists(log_dir):
    print(f"创建目录: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

# 测试写入日志
try:
    import json
    import time
    
    test_log = {
        "timestamp": int(time.time() * 1000),
        "location": "test_logging.py",
        "message": "测试日志写入",
        "data": {"test": True},
        "sessionId": "test",
        "runId": "test-1",
        "hypothesisId": "TEST"
    }
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(test_log, ensure_ascii=False) + "\n")
    
    print("日志写入成功！")
    print(f"文件存在: {os.path.exists(log_path)}")
    
    # 读取并显示日志
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"日志内容:\n{content}")
    
except Exception as e:
    print(f"日志写入失败: {e}")
    import traceback
    traceback.print_exc()

