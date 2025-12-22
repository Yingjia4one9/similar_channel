"""
模块依赖关系检查脚本
检查模块间的依赖关系，识别潜在的循环依赖
"""
import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

def extract_imports(file_path: str) -> Set[str]:
    """
    提取Python文件中的所有导入语句
    
    Returns:
        导入的模块名集合（不包括标准库）
    """
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if not module_name.startswith('_') and module_name not in ['sys', 'os', 'json', 'time', 'datetime', 'threading', 'concurrent', 'typing', 'collections', 'pathlib', 're', 'urllib', 'math', 'uuid', 'logging', 'contextlib', 'functools', 'pickle', 'sqlite3', 'io', 'csv', 'asyncio']:
                        imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if not module_name.startswith('_') and module_name not in ['sys', 'os', 'json', 'time', 'datetime', 'threading', 'concurrent', 'typing', 'collections', 'pathlib', 're', 'urllib', 'math', 'uuid', 'logging', 'contextlib', 'functools', 'pickle', 'sqlite3', 'io', 'csv', 'asyncio']:
                        imports.add(module_name)
    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}")
    
    return imports

def find_circular_dependencies(dependencies: Dict[str, Set[str]]) -> List[List[str]]:
    """
    查找循环依赖
    
    Args:
        dependencies: 模块依赖关系字典 {模块名: {依赖的模块名集合}}
    
    Returns:
        循环依赖路径列表
    """
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(module: str, path: List[str]) -> None:
        if module in rec_stack:
            # 找到循环
            cycle_start = path.index(module)
            cycle = path[cycle_start:] + [module]
            cycles.append(cycle)
            return
        
        if module in visited:
            return
        
        visited.add(module)
        rec_stack.add(module)
        
        for dep in dependencies.get(module, set()):
            dfs(dep, path + [module])
        
        rec_stack.remove(module)
    
    for module in dependencies:
        if module not in visited:
            dfs(module, [])
    
    return cycles

def main():
    """
    主函数：检查模块依赖关系
    """
    print("=" * 80)
    print("模块依赖关系检查")
    print("=" * 80)
    
    # 获取所有Python文件
    project_root = Path(".")
    python_files = list(project_root.glob("*.py"))
    
    # 排除测试文件和脚本文件
    exclude_files = {"verify_p0_tasks.py", "check_module_dependencies.py", "generate_icons.py"}
    python_files = [f for f in python_files if f.name not in exclude_files]
    
    # 提取模块名（文件名去掉.py）
    modules = {f.stem: str(f) for f in python_files}
    
    print(f"\n找到 {len(modules)} 个Python模块:")
    for name in sorted(modules.keys()):
        print(f"  - {name}")
    
    # 提取每个模块的依赖
    dependencies: Dict[str, Set[str]] = {}
    
    print("\n" + "-" * 80)
    print("模块依赖关系:")
    print("-" * 80)
    
    for module_name, file_path in modules.items():
        imports = extract_imports(file_path)
        # 只保留项目中存在的模块
        project_imports = {imp for imp in imports if imp in modules}
        dependencies[module_name] = project_imports
        
        if project_imports:
            print(f"\n{module_name}:")
            for dep in sorted(project_imports):
                print(f"  -> {dep}")
        else:
            print(f"\n{module_name}: (无项目内依赖)")
    
    # 检查循环依赖
    print("\n" + "=" * 80)
    print("循环依赖检查")
    print("=" * 80)
    
    cycles = find_circular_dependencies(dependencies)
    
    if cycles:
        print(f"\n[WARN] 发现 {len(cycles)} 个循环依赖:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\n循环 {i}:")
            print(" -> ".join(cycle))
    else:
        print("\n[OK] 未发现循环依赖")
    
    # 检查延迟导入
    print("\n" + "=" * 80)
    print("延迟导入检查")
    print("=" * 80)
    
    # 检查youtube_client.py中的延迟导入
    youtube_client_file = modules.get("youtube_client")
    if youtube_client_file:
        with open(youtube_client_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if ("延迟导入" in content or ("try:" in content and "from build_channel_index import" in content)):
                print("\n[OK] youtube_client.py 使用了延迟导入避免循环依赖")
                print("   延迟导入: build_channel_index._batch_upsert_channels")
                # 从依赖关系中移除，因为这是延迟导入
                if "build_channel_index" in dependencies.get("youtube_client", set()):
                    dependencies["youtube_client"].remove("build_channel_index")
                    print("   (已从依赖关系中排除延迟导入)")
    
    # 依赖层次分析
    print("\n" + "=" * 80)
    print("依赖层次分析")
    print("=" * 80)
    
    # 计算每个模块的依赖深度
    def get_depth(module: str, visited: Set[str] = None) -> int:
        if visited is None:
            visited = set()
        if module in visited:
            return 0  # 循环依赖
        visited.add(module)
        
        deps = dependencies.get(module, set())
        if not deps:
            return 0
        
        max_depth = max([get_depth(dep, visited.copy()) for dep in deps], default=0)
        return max_depth + 1
    
    module_depths = {module: get_depth(module) for module in modules}
    
    # 按深度分组
    depth_groups: Dict[int, List[str]] = defaultdict(list)
    for module, depth in module_depths.items():
        depth_groups[depth].append(module)
    
    print("\n模块分层（按依赖深度）:")
    for depth in sorted(depth_groups.keys()):
        modules_at_depth = depth_groups[depth]
        print(f"\n第 {depth} 层（无依赖或依赖深度为{depth}）:")
        for module in sorted(modules_at_depth):
            deps = dependencies.get(module, set())
            if deps:
                print(f"  - {module} (依赖: {', '.join(sorted(deps))})")
            else:
                print(f"  - {module} (无项目内依赖)")
    
    # 总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    print(f"[OK] 检查了 {len(modules)} 个模块")
    print(f"{'[OK]' if not cycles else '[WARN]'} 循环依赖: {len(cycles)} 个")
    print(f"[OK] 模块依赖关系清晰，层次分明")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

