"""
搜索结果缓存模块
用于缓存搜索结果，避免重复调用 API
"""
import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from threading import Lock

from infrastructure.logger import get_logger
from infrastructure.config import Config

logger = get_logger()

# 内存缓存：存储结果 ID -> 结果数据的映射
_result_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()

# 缓存过期时间（秒），默认 1 小时
CACHE_TTL = 3600

# 缓存大小限制（最大缓存项数），默认 1000
MAX_CACHE_SIZE = 1000

# 缓存统计
_cache_stats = {
    "hits": 0,  # 缓存命中次数
    "misses": 0,  # 缓存未命中次数
    "stores": 0,  # 缓存存储次数
    "evictions": 0,  # 缓存淘汰次数
}


def generate_result_id() -> str:
    """生成唯一的结果 ID"""
    return str(uuid.uuid4())


def generate_cache_key(request_params: Dict[str, Any]) -> str:
    """
    根据请求参数生成缓存键
    
    Args:
        request_params: 请求参数字典（包含所有搜索参数）
    
    Returns:
        缓存键（MD5哈希值）
    """
    # 将参数字典转换为可排序的JSON字符串，确保相同参数生成相同键
    # 排序键以确保参数顺序不影响哈希值
    sorted_params = json.dumps(request_params, sort_keys=True, ensure_ascii=False)
    # 使用MD5生成固定长度的哈希值作为缓存键
    return hashlib.md5(sorted_params.encode('utf-8')).hexdigest()


def store_result(result_id: str, result_data: Dict[str, Any], cache_key: Optional[str] = None) -> None:
    """
    存储搜索结果到缓存
    
    Args:
        result_id: 结果 ID
        result_data: 搜索结果数据
        cache_key: 可选的缓存键（用于基于参数的缓存查找）
    """
    with _cache_lock:
        # 检查缓存大小限制
        if len(_result_cache) >= MAX_CACHE_SIZE:
            # 清理最旧的过期项
            _evict_expired_items()
            
            # 如果仍然超过限制，清理最旧的项（LRU策略）
            if len(_result_cache) >= MAX_CACHE_SIZE:
                _evict_oldest_items(MAX_CACHE_SIZE // 10)  # 清理10%的旧项
                _cache_stats["evictions"] += MAX_CACHE_SIZE // 10
        
        cache_entry = {
            "data": result_data,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=CACHE_TTL),
            "access_count": 0,  # 访问次数（用于LRU）
            "last_accessed": datetime.now(),  # 最后访问时间
        }
        
        # 存储结果ID映射
        _result_cache[result_id] = cache_entry
        
        # 如果提供了缓存键，也存储缓存键到结果ID的映射
        if cache_key:
            cache_entry["cache_key"] = cache_key
            # 查找是否有相同缓存键的旧项
            for old_id, old_entry in _result_cache.items():
                if old_id != result_id and old_entry.get("cache_key") == cache_key:
                    # 删除旧项
                    del _result_cache[old_id]
                    break
        
        _cache_stats["stores"] += 1
        logger.debug(f"已缓存搜索结果: {result_id} (缓存大小: {len(_result_cache)})")


def get_result(result_id: str) -> Optional[Dict[str, Any]]:
    """
    从缓存获取搜索结果（通过结果ID）
    
    Args:
        result_id: 结果 ID
    
    Returns:
        搜索结果数据，如果不存在或已过期则返回 None
    """
    with _cache_lock:
        if result_id not in _result_cache:
            logger.debug(f"结果 ID 不存在: {result_id}")
            _cache_stats["misses"] += 1
            return None
        
        cache_entry = _result_cache[result_id]
        expires_at = cache_entry["expires_at"]
        
        # 检查是否过期
        if datetime.now() > expires_at:
            logger.debug(f"结果 ID 已过期: {result_id}")
            del _result_cache[result_id]
            _cache_stats["misses"] += 1
            return None
        
        # 更新访问统计
        cache_entry["access_count"] = cache_entry.get("access_count", 0) + 1
        cache_entry["last_accessed"] = datetime.now()
        
        _cache_stats["hits"] += 1
        logger.debug(f"从缓存获取结果: {result_id}")
        return cache_entry["data"]


def get_result_by_params(request_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    根据请求参数从缓存获取搜索结果
    
    Args:
        request_params: 请求参数字典
    
    Returns:
        搜索结果数据，如果不存在或已过期则返回 None
    """
    cache_key = generate_cache_key(request_params)
    
    with _cache_lock:
        # 查找具有相同缓存键的项
        for result_id, cache_entry in _result_cache.items():
            if cache_entry.get("cache_key") == cache_key:
                expires_at = cache_entry["expires_at"]
                
                # 检查是否过期
                if datetime.now() > expires_at:
                    logger.debug(f"缓存键已过期: {cache_key}")
                    del _result_cache[result_id]
                    _cache_stats["misses"] += 1
                    return None
                
                # 更新访问统计
                cache_entry["access_count"] = cache_entry.get("access_count", 0) + 1
                cache_entry["last_accessed"] = datetime.now()
                
                _cache_stats["hits"] += 1
                logger.debug(f"从缓存获取结果（通过参数）: {cache_key}")
                return cache_entry["data"]
        
        _cache_stats["misses"] += 1
        logger.debug(f"缓存键不存在: {cache_key}")
        return None


def delete_result(result_id: str) -> bool:
    """
    删除缓存的结果
    
    Args:
        result_id: 结果 ID
    
    Returns:
        是否成功删除
    """
    with _cache_lock:
        if result_id in _result_cache:
            del _result_cache[result_id]
            logger.debug(f"已删除缓存结果: {result_id}")
            return True
        return False


def cleanup_expired() -> int:
    """
    清理过期的缓存项
    
    Returns:
        清理的数量
    """
    now = datetime.now()
    expired_ids = []
    
    with _cache_lock:
        for result_id, cache_entry in _result_cache.items():
            if now > cache_entry["expires_at"]:
                expired_ids.append(result_id)
        
        for result_id in expired_ids:
            del _result_cache[result_id]
    
    if expired_ids:
        logger.debug(f"清理了 {len(expired_ids)} 个过期的缓存项")
    
    return len(expired_ids)


def get_cache_size() -> int:
    """获取当前缓存大小"""
    with _cache_lock:
        return len(_result_cache)


def get_cache_stats() -> Dict[str, Any]:
    """
    获取缓存统计信息
    
    Returns:
        包含缓存统计信息的字典
    """
    with _cache_lock:
        total_requests = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = (_cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(_result_cache),
            "max_size": MAX_CACHE_SIZE,
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "stores": _cache_stats["stores"],
            "evictions": _cache_stats["evictions"],
            "hit_rate": round(hit_rate, 2),
            "ttl_seconds": CACHE_TTL,
        }


def _evict_expired_items() -> int:
    """清理过期的缓存项（内部函数）"""
    now = datetime.now()
    expired_ids = []
    
    for result_id, cache_entry in _result_cache.items():
        if now > cache_entry["expires_at"]:
            expired_ids.append(result_id)
    
    for result_id in expired_ids:
        del _result_cache[result_id]
    
    return len(expired_ids)


def _evict_oldest_items(count: int) -> int:
    """
    清理最旧的缓存项（LRU策略，内部函数）
    
    Args:
        count: 要清理的数量
    
    Returns:
        实际清理的数量
    """
    if count <= 0 or len(_result_cache) == 0:
        return 0
    
    # 按最后访问时间排序，选择最旧的项
    sorted_items = sorted(
        _result_cache.items(),
        key=lambda x: x[1].get("last_accessed", x[1].get("created_at", datetime.min))
    )
    
    evicted = 0
    for result_id, _ in sorted_items[:count]:
        if result_id in _result_cache:
            del _result_cache[result_id]
            evicted += 1
    
    return evicted

