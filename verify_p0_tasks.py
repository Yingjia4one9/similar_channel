"""
P0级别任务验证脚本
验证以下检查点：
- CP-y2-01：FAISS索引使用验证
- CP-y2-04：本地索引优先使用验证
- CP-y5-06：SQL注入防护验证
"""
import ast
import os
import re
from typing import List, Tuple, Dict
from pathlib import Path

def find_sql_executions(file_path: str) -> List[Tuple[int, str, str]]:
    """
    查找文件中的所有SQL执行语句
    
    Returns:
        List of (line_number, code_line, context) tuples
    """
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # 查找 execute 或 executemany 调用
            if re.search(r'\.(execute|executemany)\s*\(', line):
                # 获取上下文（前后各2行）
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 2)
                context = ''.join(lines[context_start:context_end])
                results.append((i, line.strip(), context))
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    
    return results

def check_sql_injection_protection(file_path: str) -> Dict:
    """
    检查SQL注入防护：验证所有SQL查询是否使用参数化查询
    
    Returns:
        验证结果字典
    """
    results = {
        'file': file_path,
        'total_executions': 0,
        'parameterized': 0,
        'potential_risks': [],
        'safe_patterns': []
    }
    
    sql_executions = find_sql_executions(file_path)
    results['total_executions'] = len(sql_executions)
    
    for line_num, code_line, context in sql_executions:
        # 检查是否使用参数化查询
        # 安全的模式：使用 ? 占位符或命名参数
        has_placeholder = '?' in code_line or re.search(r':\w+', code_line)
        
        # 检查是否有字符串格式化（危险）
        has_f_string = 'f"' in code_line or "f'" in code_line
        has_format = '.format(' in code_line
        has_percent = '%' in code_line and ('%s' in code_line or '%d' in code_line)
        
        # 检查是否有字符串拼接（可能危险）
        # 但如果是构建占位符（如 IN 子句），且参数是参数化的，则是安全的
        has_string_concat = '+' in code_line and ('"' in code_line or "'" in code_line)
        
        # 检查是否是安全的占位符构建模式
        # 例如：placeholders = ",".join(["?" for _ in channel_ids])
        is_safe_placeholder_build = (
            'placeholders' in code_line.lower() or
            'join' in code_line.lower() and '?' in code_line
        )
        
        if has_placeholder:
            results['parameterized'] += 1
            if is_safe_placeholder_build:
                results['safe_patterns'].append({
                    'line': line_num,
                    'code': code_line,
                    'reason': '安全的占位符构建（用于IN子句）'
                })
            else:
                results['safe_patterns'].append({
                    'line': line_num,
                    'code': code_line,
                    'reason': '使用参数化查询'
                })
        elif has_f_string or has_format or has_percent:
            results['potential_risks'].append({
                'line': line_num,
                'code': code_line,
                'context': context,
                'risk': '使用字符串格式化，可能存在SQL注入风险'
            })
        elif has_string_concat and not is_safe_placeholder_build:
            # 需要进一步检查上下文
            results['potential_risks'].append({
                'line': line_num,
                'code': code_line,
                'context': context,
                'risk': '使用字符串拼接，需要检查是否安全'
            })
        else:
            # 可能是DDL语句（CREATE TABLE等），这些通常是安全的
            if any(keyword in code_line.upper() for keyword in ['CREATE', 'DROP', 'ALTER', 'INDEX']):
                results['safe_patterns'].append({
                    'line': line_num,
                    'code': code_line,
                    'reason': 'DDL语句，通常安全'
                })
            else:
                results['potential_risks'].append({
                    'line': line_num,
                    'code': code_line,
                    'context': context,
                    'risk': '未检测到参数化查询模式，需要人工检查'
                })
    
    return results

def check_faiss_usage() -> Dict:
    """
    检查FAISS索引使用情况
    """
    results = {
        'faiss_imported': False,
        'faiss_available_check': False,
        'faiss_usage_locations': [],
        'fallback_mechanism': False,
        'issues': []
    }
    
    database_file = 'database.py'
    if not os.path.exists(database_file):
        results['issues'].append(f'文件 {database_file} 不存在')
        return results
    
    with open(database_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 检查FAISS导入
    if 'import faiss' in content or 'from faiss' in content:
        results['faiss_imported'] = True
    
    # 检查FAISS_AVAILABLE
    if 'FAISS_AVAILABLE' in content:
        results['faiss_available_check'] = True
    
    # 查找FAISS使用位置
    for i, line in enumerate(lines, 1):
        if 'faiss.' in line.lower() or 'FAISS_AVAILABLE' in line:
            results['faiss_usage_locations'].append({
                'line': i,
                'code': line.strip()
            })
    
    # 检查回退机制
    if 'if FAISS_AVAILABLE' in content or 'if not FAISS_AVAILABLE' in content:
        results['fallback_mechanism'] = True
    
    # 检查回退逻辑
    if '回退到原始方法' in content or 'fallback' in content.lower() or '回退' in content:
        results['fallback_mechanism'] = True
    
    if not results['faiss_imported']:
        results['issues'].append('未找到FAISS导入语句')
    if not results['faiss_available_check']:
        results['issues'].append('未找到FAISS_AVAILABLE检查')
    if not results['fallback_mechanism']:
        results['issues'].append('未找到回退机制')
    
    return results

def check_local_index_priority() -> Dict:
    """
    检查本地索引优先使用逻辑
    """
    results = {
        'local_index_queries': [],
        'api_calls_after_local': [],
        'priority_logic_correct': False,
        'issues': []
    }
    
    youtube_client_file = 'youtube_client.py'
    if not os.path.exists(youtube_client_file):
        results['issues'].append(f'文件 {youtube_client_file} 不存在')
        return results
    
    with open(youtube_client_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 查找本地索引查询
    for i, line in enumerate(lines, 1):
        if 'get_candidates_from_local_index' in line:
            results['local_index_queries'].append({
                'line': i,
                'code': line.strip()
            })
        if 'get_channel_info_from_local_db' in line:
            results['local_index_queries'].append({
                'line': i,
                'code': line.strip()
            })
        if 'batch_get_channels_info' in line:
            results['api_calls_after_local'].append({
                'line': i,
                'code': line.strip()
            })
    
    # 检查优先级逻辑：本地查询应该在API调用之前
    # 简单检查：如果找到了本地查询和API调用，且本地查询的行号小于API调用，则认为逻辑正确
    if results['local_index_queries'] and results['api_calls_after_local']:
        local_lines = [q['line'] for q in results['local_index_queries']]
        api_lines = [q['line'] for q in results['api_calls_after_local']]
        if min(local_lines) < min(api_lines):
            results['priority_logic_correct'] = True
        else:
            results['issues'].append('API调用可能在本地查询之前，需要检查逻辑顺序')
    
    # 检查是否有"缺失的"或"missing"的逻辑
    if 'missing' in content.lower() or '缺失' in content:
        results['priority_logic_correct'] = True
    
    if not results['local_index_queries']:
        results['issues'].append('未找到本地索引查询')
    if not results['api_calls_after_local']:
        results['issues'].append('未找到API调用（可能所有数据都从本地获取）')
    
    return results

def main():
    """
    主验证函数
    """
    print("=" * 80)
    print("P0级别任务验证报告")
    print("=" * 80)
    
    # 1. 验证 CP-y2-01：FAISS索引使用
    print("\n【CP-y2-01】FAISS索引使用验证")
    print("-" * 80)
    faiss_results = check_faiss_usage()
    print(f"[OK] FAISS已导入: {faiss_results['faiss_imported']}")
    print(f"[OK] FAISS可用性检查: {faiss_results['faiss_available_check']}")
    print(f"[OK] 回退机制: {faiss_results['fallback_mechanism']}")
    print(f"[OK] FAISS使用位置数: {len(faiss_results['faiss_usage_locations'])}")
    if faiss_results['issues']:
        print(f"[WARN] 发现问题:")
        for issue in faiss_results['issues']:
            print(f"  - {issue}")
    else:
        print("[OK] 所有检查通过")
    
    # 2. 验证 CP-y2-04：本地索引优先使用
    print("\n【CP-y2-04】本地索引优先使用验证")
    print("-" * 80)
    local_index_results = check_local_index_priority()
    print(f"[OK] 本地索引查询位置数: {len(local_index_results['local_index_queries'])}")
    print(f"[OK] API调用位置数: {len(local_index_results['api_calls_after_local'])}")
    print(f"[OK] 优先级逻辑正确: {local_index_results['priority_logic_correct']}")
    if local_index_results['issues']:
        print(f"[WARN] 发现问题:")
        for issue in local_index_results['issues']:
            print(f"  - {issue}")
    else:
        print("[OK] 所有检查通过")
    
    # 3. 验证 CP-y5-06：SQL注入防护
    print("\n【CP-y5-06】SQL注入防护验证")
    print("-" * 80)
    
    files_to_check = [
        'database.py',
        'build_channel_index.py',
        'youtube_client.py',
        'quota_tracker.py',
        'database_utils.py'
    ]
    
    all_sql_results = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n检查文件: {file_path}")
            sql_results = check_sql_injection_protection(file_path)
            all_sql_results.append(sql_results)
            
            print(f"  总SQL执行数: {sql_results['total_executions']}")
            print(f"  参数化查询数: {sql_results['parameterized']}")
            print(f"  安全模式数: {len(sql_results['safe_patterns'])}")
            print(f"  潜在风险数: {len(sql_results['potential_risks'])}")
            
            if sql_results['potential_risks']:
                print(f"  [WARN] 潜在风险:")
                for risk in sql_results['potential_risks']:
                    print(f"    行 {risk['line']}: {risk['risk']}")
                    print(f"    代码: {risk['code']}")
            else:
                print(f"  [OK] 未发现SQL注入风险")
        else:
            print(f"  文件不存在: {file_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    # FAISS验证总结
    faiss_ok = (
        faiss_results['faiss_imported'] and
        faiss_results['faiss_available_check'] and
        faiss_results['fallback_mechanism'] and
        len(faiss_results['issues']) == 0
    )
    print(f"CP-y2-01 (FAISS索引使用): {'[PASS] 通过' if faiss_ok else '[WARN] 需要检查'}")
    
    # 本地索引验证总结
    local_index_ok = (
        len(local_index_results['local_index_queries']) > 0 and
        local_index_results['priority_logic_correct'] and
        len(local_index_results['issues']) == 0
    )
    print(f"CP-y2-04 (本地索引优先): {'[PASS] 通过' if local_index_ok else '[WARN] 需要检查'}")
    
    # SQL注入防护总结
    sql_ok = all(
        len(r['potential_risks']) == 0 for r in all_sql_results
    )
    print(f"CP-y5-06 (SQL注入防护): {'[PASS] 通过' if sql_ok else '[WARN] 需要检查'}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

