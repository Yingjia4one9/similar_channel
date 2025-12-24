"""
缓存模块
提供内存缓存功能，减少重复的API调用
"""
import json
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from infrastructure.logger import get_logger
from infrastructure.config import Config

logger = get_logger()

# 类型变量
F = TypeVar('F', bound=Callable[..., Any])


class SimpleCache:
    """简单的内存缓存实现（线程安全）"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        初始化缓存
        
        Args:
            default_ttl: 默认过期时间（秒），默认1小时
        """
        self._cache: Dict[str, tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值（线程安全）
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值，如果不存在或已过期则返回 None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expire_time = self._cache[key]
            if time.time() > expire_time:
                # 已过期，删除
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值（线程安全）
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），如果为 None 则使用默认值
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            expire_time = time.time() + ttl
            self._cache[key] = (value, expire_time)
    
    def clear(self) -> None:
        """清空所有缓存（线程安全）"""
        with self._lock:
            self._cache.clear()
        logger.info("缓存已清空")
    
    def remove(self, key: str) -> None:
        """移除指定键的缓存（线程安全）"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def size(self) -> int:
        """返回缓存项数量（线程安全）"""
        with self._lock:
            # 清理过期项
            current_time = time.time()
            expired_keys = [
                key for key, (_, expire_time) in self._cache.items()
                if current_time > expire_time
            ]
            for key in expired_keys:
                del self._cache[key]
            
            return len(self._cache)


# 全局缓存实例（使用配置的TTL）
_channel_info_cache = SimpleCache(
    default_ttl=Config.get_config_value("cache.ttl.channel_info", Config.CACHE_TTL["channel_info"])
)
_channel_videos_cache = SimpleCache(
    default_ttl=Config.get_config_value("cache.ttl.channel_videos", Config.CACHE_TTL["channel_videos"])
)


def cached_channel_info(ttl: int | None = None):
    """
    装饰器：缓存频道信息
    
    Args:
        ttl: 缓存过期时间（秒），如果为None则使用配置的默认值
    """
    if ttl is None:
        ttl = Config.get_config_value("cache.ttl.channel_info", Config.CACHE_TTL["channel_info"])
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(channel_id: str, *args, **kwargs):
            cache_key = f"channel_info:{channel_id}"
            cached_value = _channel_info_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"从缓存获取频道信息: {channel_id}")
                return cached_value
            
            result = func(channel_id, *args, **kwargs)
            _channel_info_cache.set(cache_key, result, ttl=ttl)
            return result
        
        return wrapper  # type: ignore
    return decorator


def cached_channel_videos(ttl: int | None = None):
    """
    装饰器：缓存频道视频列表
    
    Args:
        ttl: 缓存过期时间（秒），如果为None则使用配置的默认值
    """
    if ttl is None:
        ttl = Config.get_config_value("cache.ttl.channel_videos", Config.CACHE_TTL["channel_videos"])
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(channel_id: str, *args, **kwargs):
            max_results = kwargs.get('max_results', 5)
            cache_key = f"channel_videos:{channel_id}:{max_results}"
            cached_value = _channel_videos_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"从缓存获取频道视频: {channel_id}")
                return cached_value
            
            result = func(channel_id, *args, **kwargs)
            _channel_videos_cache.set(cache_key, result, ttl=ttl)
            return result
        
        return wrapper  # type: ignore
    return decorator


def clear_all_caches() -> None:
    """清空所有缓存"""
    _channel_info_cache.clear()
    _channel_videos_cache.clear()


def invalidate_channel_info_cache(channel_id: str) -> None:
    """
    失效指定频道的缓存
    
    Args:
        channel_id: 频道 ID
    """
    cache_key = f"channel_info:{channel_id}"
    _channel_info_cache.remove(cache_key)
    logger.debug(f"已失效频道信息缓存: {channel_id}")


def invalidate_channel_videos_cache(channel_id: str, max_results: int | None = None) -> None:
    """
    失效指定频道的视频列表缓存（线程安全，修复竞态条件）
    
    Args:
        channel_id: 频道 ID
        max_results: 视频数量，如果为None则失效所有相关缓存
    """
    import time
    import json
    log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
    
    if max_results is not None:
        cache_key = f"channel_videos:{channel_id}:{max_results}"
        _channel_videos_cache.remove(cache_key)
        logger.debug(f"已失效频道视频缓存: {channel_id} (max_results={max_results})")
    else:
        # 失效所有该频道的视频缓存（修复竞态条件：在锁内完成所有操作）
        prefix = f"channel_videos:{channel_id}:"
        keys_to_remove = []
        log_path = r"c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log"
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "cache.py:invalidate_channel_videos_cache",
                    "message": "开始失效缓存（修复竞态条件）",
                    "data": {"channel_id": channel_id, "prefix": prefix},
                    "sessionId": "debug-session",
                    "runId": "fix-verification",
                    "hypothesisId": "B"
                }, ensure_ascii=False) + "\n")
        except: pass
        # #endregion
        
        # 在锁内获取所有需要删除的键并立即删除，避免竞态条件
        with _channel_videos_cache._lock:
            # 创建键的副本列表，避免在迭代时修改字典
            all_keys = list(_channel_videos_cache._cache.keys())
            for key in all_keys:
                if key.startswith(prefix):
                    keys_to_remove.append(key)
            
            # 在锁内删除所有匹配的键
            for key in keys_to_remove:
                if key in _channel_videos_cache._cache:
                    del _channel_videos_cache._cache[key]
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "cache.py:invalidate_channel_videos_cache",
                    "message": "缓存失效完成",
                    "data": {"channel_id": channel_id, "removed_count": len(keys_to_remove)},
                    "sessionId": "debug-session",
                    "runId": "fix-verification",
                    "hypothesisId": "B"
                }, ensure_ascii=False) + "\n")
        except: pass
        # #endregion
        
        if keys_to_remove:
            logger.debug(f"已失效频道所有视频缓存: {channel_id} (共{len(keys_to_remove)}项)")


def invalidate_all_channel_caches(channel_id: str) -> None:
    """
    失效指定频道的所有缓存（信息和视频列表）
    
    Args:
        channel_id: 频道 ID
    """
    invalidate_channel_info_cache(channel_id)
    invalidate_channel_videos_cache(channel_id)
    logger.info(f"已失效频道所有缓存: {channel_id}")
