"""
YouTube API 调用模块
处理 API 请求、重试逻辑和错误处理
"""
import asyncio
import threading
from typing import Any, Dict

import httpx

from infrastructure.config import Config
from infrastructure.logger import get_logger
from infrastructure.quota_tracker import record_quota_usage, check_and_update_rate_limit, get_rate_limit_status

logger = get_logger()


class YouTubeQuotaExceededError(Exception):
    """YouTube API 配额已用完的异常"""
    pass


class YouTubeAPIError(Exception):
    """YouTube API 通用错误"""
    pass


class YouTubeAPIClient:
    """YouTube API 客户端（异步版本）"""
    
    BASE_URL = "https://www.googleapis.com/youtube/v3/"
    MAX_RETRIES = 3
    
    def __init__(self, api_key: str | None = None, use_for: str | None = None):
        """
        初始化 API 客户端
        
        Args:
            api_key: API Key，如果为 None 则从配置加载
            use_for: API Key 用途标识（"index" 或 "search"），用于配额跟踪
        """
        self._api_key = api_key or Config.load_api_key()
        self._use_for = use_for
        # 使用 httpx.AsyncClient，支持连接池和异步请求
        self._client: httpx.AsyncClient | None = None
        # 存储客户端创建时的事件循环 ID，用于检测事件循环变化
        self._client_loop_id: int | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步HTTP客户端（延迟初始化）
        
        注意：httpx.AsyncClient 绑定到创建它的事件循环。
        如果事件循环发生变化（例如在不同线程中使用），需要重新创建客户端。
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            # 如果没有运行中的事件循环，使用 None
            current_loop_id = None
        
        # 如果客户端不存在，或者事件循环发生了变化，重新创建客户端
        if self._client is None or self._client_loop_id != current_loop_id:
            # 如果已有客户端，先关闭它
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception:
                    pass  # 忽略关闭错误
            
            timeout = Config.get_config_value("API_TIMEOUT", Config.API_TIMEOUT, "YT_API_TIMEOUT")
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
            self._client_loop_id = current_loop_id
        
        return self._client
    
    async def close(self):
        """关闭HTTP客户端（清理资源）"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def get(self, endpoint: str, params: Dict[str, Any], max_retries: int | None = None) -> Dict[str, Any]:
        """
        调用 YouTube API，带重试机制（异步版本）。
        
        Args:
            endpoint: API 端点（如 "search", "channels"）
            params: 请求参数
            max_retries: 最大重试次数（默认使用类常量）
        
        Returns:
            API 响应的 JSON 数据
            
        Raises:
            YouTubeQuotaExceededError: API 配额已用完
            ConnectionError: 网络连接错误
            TimeoutError: 请求超时
            YouTubeAPIError: 其他 API 错误
        """
        max_retries = max_retries or self.MAX_RETRIES
        # API Key通过URL参数传递（YouTube Data API v3的要求）
        # 注意：虽然API Key在URL中，但通过HTTPS传输是安全的
        # 确保不在日志中记录完整的URL（包含API Key）
        all_params = {"key": self._api_key, **params}
        
        # 检查配额限流状态（CP-y3-05：配额限流机制）
        from infrastructure.quota_tracker import DEFAULT_DAILY_QUOTA
        is_rate_limited, delay_seconds = check_and_update_rate_limit(
            daily_quota=DEFAULT_DAILY_QUOTA, 
            use_for=self._use_for
        )
        if is_rate_limited and delay_seconds > 0:
            logger.debug(
                f"[{self._use_for or 'default'}] 配额限流生效，延迟 {delay_seconds:.2f} 秒后执行API调用"
            )
            await asyncio.sleep(delay_seconds)
        
        client = await self._get_client()
        last_exception = None
        quota_recorded = False  # 标记是否已记录配额使用
        for attempt in range(max_retries):
            try:
                resp = await client.get(endpoint, params=all_params)
                resp.raise_for_status()
                result = resp.json()
                # 只在最终成功时记录配额使用（避免重试时重复记录）
                if not quota_recorded:
                    record_quota_usage(
                        endpoint=endpoint,
                        method="list",
                        params=params,
                        success=True,
                        use_for=self._use_for,
                    )
                    quota_recorded = True
                return result
            except httpx.ConnectError as e:
                # DNS 解析失败或网络连接错误
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
                    error_msg = str(e)
                    if "getaddrinfo failed" in error_msg or "Failed to resolve" in error_msg:
                        logger.warning(f"无法连接到 YouTube API (DNS 解析失败)，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    else:
                        logger.warning(f"连接失败，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise ConnectionError(
                        f"无法连接到 YouTube API。请检查网络连接和 DNS 设置。\n"
                        f"错误详情: {e}\n"
                        f"如果问题持续存在，可能是网络配置或防火墙问题。"
                    ) from e
            except httpx.TimeoutException as e:
                # 请求超时
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"请求超时，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise TimeoutError(
                        f"YouTube API 请求超时。请检查网络连接或稍后重试。\n"
                        f"错误详情: {e}"
                    ) from e
            except httpx.HTTPStatusError as e:
                # HTTP 错误（如 400, 401, 403, 429, 500 等）
                status_code = e.response.status_code if e.response is not None else None
                
                # 尝试解析错误信息
                message = None
                try:
                    if e.response is not None:
                        err_json = e.response.json()
                        message = err_json.get("error", {}).get("message", "")
                except Exception:
                    pass
                
                # 检查是否是配额错误（403 且错误信息包含 "quota"）
                if status_code == 403 and message and "quota" in message.lower():
                    # 记录配额耗尽事件（但不记录配额消耗，因为请求未成功）
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            cost=0,  # 配额耗尽时不消耗配额
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                    raise YouTubeQuotaExceededError(
                        f"YouTube API 配额已用完。\n"
                        f"错误信息: {message}\n\n"
                        f"解决方案：\n"
                        f"1. 等待配额重置（通常在每天UTC 00:00重置）\n"
                        f"2. 使用本地索引数据（如果已构建）\n"
                        f"3. 申请更高的API配额或使用多个API Key"
                    ) from e
                
                # 429 (Too Many Requests) 或 5xx 服务器错误，可以重试
                if status_code == 429 or (status_code and 500 <= status_code < 600):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"HTTP {status_code} 服务器错误，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    # 如果所有重试都失败，记录配额使用
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                
                # 其他 HTTP 错误，记录失败但抛出异常
                if not quota_recorded:
                    record_quota_usage(
                        endpoint=endpoint,
                        method="list",
                        params=params,
                        success=False,
                        use_for=self._use_for,
                    )
                    quota_recorded = True
                if message:
                    raise YouTubeAPIError(f"YouTube API 错误: {message}") from e
                raise
            except RuntimeError as e:
                # 处理特定的 RuntimeError
                error_msg = str(e) if e else f"{type(e).__name__}"
                # 如果客户端已关闭，重新创建并重试（使用更宽松的匹配）
                if "client has been closed" in error_msg.lower() or "Cannot send a request" in error_msg:
                    if attempt < max_retries - 1:
                        # 重新创建客户端
                        try:
                            self._client = None
                            client = await self._get_client()
                            wait_time = 2 ** attempt
                            logger.warning(f"客户端已关闭，重新创建后 {wait_time} 秒重试 ({attempt + 1}/{max_retries})...")
                            await asyncio.sleep(wait_time)
                            continue
                        except (RuntimeError, Exception) as recreate_error:
                            # 如果重新创建客户端失败（可能是事件循环已关闭或绑定错误），检查是否是关闭错误
                            recreate_error_msg = str(recreate_error).lower()
                            if ("closed" in recreate_error_msg or 
                                "shutdown" in recreate_error_msg or 
                                "bound to a different" in recreate_error_msg):
                                if not quota_recorded:
                                    record_quota_usage(
                                        endpoint=endpoint,
                                        method="list",
                                        params=params,
                                        success=False,
                                        use_for=self._use_for,
                                    )
                                    quota_recorded = True
                                raise
                            # 其他错误也抛出
                            raise
                    else:
                        # 最后一次重试也失败
                        if not quota_recorded:
                            record_quota_usage(
                                endpoint=endpoint,
                                method="list",
                                params=params,
                                success=False,
                                use_for=self._use_for,
                            )
                            quota_recorded = True
                        raise
                
                # 如果事件循环已关闭、解释器正在关闭，或者事件循环绑定错误，直接抛出，不要重试
                is_shutdown_error = (
                    "Event loop is closed" in error_msg or 
                    "cannot schedule new futures" in error_msg or
                    "interpreter shutdown" in error_msg or
                    "shutdown" in error_msg.lower() or
                    "bound to a different event loop" in error_msg
                )
                if is_shutdown_error:
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                    raise
                
                # 其他 RuntimeError 作为普通异常处理
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    error_str = str(e) if e else f"{type(e).__name__}"
                    logger.warning(f"发生未预期的错误，{wait_time} 秒后重试 ({attempt + 1}/{max_retries}): {error_str}")
                    try:
                        await asyncio.sleep(wait_time)
                    except RuntimeError as sleep_error:
                        # 如果 sleep 时事件循环已关闭，直接抛出
                        if "closed" in str(sleep_error).lower() or "shutdown" in str(sleep_error).lower():
                            if not quota_recorded:
                                record_quota_usage(
                                    endpoint=endpoint,
                                    method="list",
                                    params=params,
                                    success=False,
                                    use_for=self._use_for,
                                )
                                quota_recorded = True
                            raise
                        raise
                else:
                    # 所有重试都失败，记录配额使用
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                    raise
            except Exception as e:
                # 其他未预期的错误
                # 如果事件循环已关闭或解释器正在关闭，直接抛出，不要重试（重试需要 sleep，而 sleep 需要事件循环）
                error_msg = str(e)
                is_shutdown_error = (
                    "Event loop is closed" in error_msg or 
                    "cannot schedule new futures" in error_msg or
                    "interpreter shutdown" in error_msg or
                    "bound to a different event loop" in error_msg or
                    (isinstance(e, RuntimeError) and ("closed" in error_msg.lower() or "shutdown" in error_msg.lower() or "bound to a different" in error_msg.lower()))
                )
                if is_shutdown_error:
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                    raise
                
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    error_str = str(e) if e else f"{type(e).__name__}"
                    logger.warning(f"发生未预期的错误，{wait_time} 秒后重试 ({attempt + 1}/{max_retries}): {error_str}")
                    try:
                        await asyncio.sleep(wait_time)
                    except RuntimeError as sleep_error:
                        # 如果 sleep 时事件循环已关闭，直接抛出
                        if "closed" in str(sleep_error).lower() or "shutdown" in str(sleep_error).lower():
                            if not quota_recorded:
                                record_quota_usage(
                                    endpoint=endpoint,
                                    method="list",
                                    params=params,
                                    success=False,
                                    use_for=self._use_for,
                                )
                                quota_recorded = True
                            raise
                        raise
                else:
                    # 所有重试都失败，记录配额使用
                    if not quota_recorded:
                        record_quota_usage(
                            endpoint=endpoint,
                            method="list",
                            params=params,
                            success=False,
                            use_for=self._use_for,
                        )
                        quota_recorded = True
                    raise
        
        # 如果所有重试都失败了
        if last_exception:
            # 记录最终失败（如果还没记录）
            if not quota_recorded:
                record_quota_usage(
                    endpoint=endpoint,
                    method="list",
                    params=params,
                    success=False,
                    use_for=self._use_for,
                )
            raise last_exception
        # 记录失败（如果还没记录）
        if not quota_recorded:
            record_quota_usage(
                endpoint=endpoint,
                method="list",
                params=params,
                success=False,
                use_for=self._use_for,
            )
        raise RuntimeError("请求失败，原因未知")


# 全局 API 客户端实例（延迟初始化）
# 注意：在多线程环境中，每个线程应该有独立的客户端实例
# 使用线程本地存储来确保线程安全
_thread_local = threading.local()

_api_client: YouTubeAPIClient | None = None
_api_client_index: YouTubeAPIClient | None = None  # 索引构建专用（主线程）
_api_client_search: YouTubeAPIClient | None = None  # 实时搜索专用（主线程）


async def close_all_clients():
    """关闭所有全局 API 客户端（用于清理资源）"""
    global _api_client, _api_client_index, _api_client_search
    clients = [_api_client, _api_client_index, _api_client_search]
    for client in clients:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass  # 忽略关闭错误


async def get_api_client(use_for: str | None = None) -> YouTubeAPIClient:
    """
    获取 API 客户端实例（线程安全版本）
    
    在多线程环境中，每个线程都有独立的客户端实例，避免客户端被意外关闭。
    
    Args:
        use_for: 用途标识，可选值：
            - "index": 索引构建专用
            - "search": 实时搜索专用
            - None: 默认（向后兼容）
    
    Returns:
        YouTubeAPIClient 实例
    """
    global _api_client, _api_client_index, _api_client_search
    
    # 获取当前线程的客户端存储
    if not hasattr(_thread_local, 'clients'):
        _thread_local.clients = {}
    
    # 生成客户端键
    client_key = use_for or 'default'
    
    # 如果当前线程已有客户端，直接返回
    if client_key in _thread_local.clients:
        return _thread_local.clients[client_key]
    
    # 为当前线程创建新的客户端实例
    if use_for == "index":
        api_key = Config.load_api_key_for_index()
        client = YouTubeAPIClient(api_key=api_key, use_for="index")
        logger.debug(f"初始化索引构建专用API客户端（线程: {threading.current_thread().name}）")
    elif use_for == "search":
        api_key = Config.load_api_key_for_search()
        client = YouTubeAPIClient(api_key=api_key, use_for="search")
        logger.debug(f"初始化实时搜索专用API客户端（线程: {threading.current_thread().name}）")
    else:
        # 默认客户端（向后兼容）
        client = YouTubeAPIClient(use_for=None)
        logger.debug(f"初始化默认API客户端（线程: {threading.current_thread().name}）")
    
    # 保存到线程本地存储
    _thread_local.clients[client_key] = client
    
    # 同时更新全局变量（用于主线程，保持向后兼容）
    if threading.current_thread() is threading.main_thread():
        if use_for == "index":
            _api_client_index = client
        elif use_for == "search":
            _api_client_search = client
        else:
            _api_client = client
    
    return client


async def yt_get(
    endpoint: str, 
    params: Dict[str, Any], 
    max_retries: int | None = None,
    use_for: str | None = None
) -> Dict[str, Any]:
    """
    便捷函数：调用 YouTube API（异步版本）
    
    Args:
        endpoint: API 端点
        params: 请求参数
        max_retries: 最大重试次数
        use_for: 用途标识，可选值：
            - "index": 索引构建专用
            - "search": 实时搜索专用
            - None: 默认（向后兼容）
    
    Returns:
        API 响应的 JSON 数据
    """
    client = await get_api_client(use_for=use_for)
    return await client.get(endpoint, params, max_retries)

