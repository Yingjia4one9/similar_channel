"""
YouTube API 调用模块
处理 API 请求、重试逻辑和错误处理
"""
import time
from typing import Any, Dict

import requests

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
    """YouTube API 客户端"""
    
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
    
    def get(self, endpoint: str, params: Dict[str, Any], max_retries: int | None = None) -> Dict[str, Any]:
        """
        调用 YouTube API，带重试机制。
        
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
        base_url = self.BASE_URL + endpoint
        # API Key通过URL参数传递（YouTube Data API v3的要求）
        # 注意：虽然API Key在URL中，但通过HTTPS传输是安全的
        # 确保不在日志中记录完整的URL（包含API Key）
        all_params = {"key": self._api_key, **params}
        
        # 从配置获取超时时间（支持环境变量覆盖）
        timeout = Config.get_config_value("API_TIMEOUT", Config.API_TIMEOUT, "YT_API_TIMEOUT")
        
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
            time.sleep(delay_seconds)
        
        last_exception = None
        quota_recorded = False  # 标记是否已记录配额使用
        for attempt in range(max_retries):
            try:
                resp = requests.get(base_url, params=all_params, timeout=timeout)
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
            except requests.exceptions.ConnectionError as e:
                # DNS 解析失败或网络连接错误
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
                    error_msg = str(e)
                    if "getaddrinfo failed" in error_msg or "Failed to resolve" in error_msg:
                        logger.warning(f"无法连接到 YouTube API (DNS 解析失败)，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    else:
                        logger.warning(f"连接失败，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError(
                        f"无法连接到 YouTube API。请检查网络连接和 DNS 设置。\n"
                        f"错误详情: {e}\n"
                        f"如果问题持续存在，可能是网络配置或防火墙问题。"
                    ) from e
            except requests.exceptions.Timeout as e:
                # 请求超时
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"请求超时，{wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise TimeoutError(
                        f"YouTube API 请求超时。请检查网络连接或稍后重试。\n"
                        f"错误详情: {e}"
                    ) from e
            except requests.HTTPError as e:
                # HTTP 错误（如 400, 401, 403, 429, 500 等）
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None
                
                # 尝试解析错误信息
                message = None
                try:
                    if hasattr(e, 'response') and e.response is not None:
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
                        time.sleep(wait_time)
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
            except Exception as e:
                # 其他未预期的错误
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"发生未预期的错误，{wait_time} 秒后重试 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
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
_api_client: YouTubeAPIClient | None = None
_api_client_index: YouTubeAPIClient | None = None  # 索引构建专用
_api_client_search: YouTubeAPIClient | None = None  # 实时搜索专用


def get_api_client(use_for: str | None = None) -> YouTubeAPIClient:
    """
    获取 API 客户端实例
    
    Args:
        use_for: 用途标识，可选值：
            - "index": 索引构建专用
            - "search": 实时搜索专用
            - None: 默认（向后兼容）
    
    Returns:
        YouTubeAPIClient 实例
    """
    global _api_client, _api_client_index, _api_client_search
    
    if use_for == "index":
        if _api_client_index is None:
            api_key = Config.load_api_key_for_index()
            _api_client_index = YouTubeAPIClient(api_key=api_key, use_for="index")
            logger.debug("初始化索引构建专用API客户端")
        return _api_client_index
    elif use_for == "search":
        if _api_client_search is None:
            api_key = Config.load_api_key_for_search()
            _api_client_search = YouTubeAPIClient(api_key=api_key, use_for="search")
            logger.debug("初始化实时搜索专用API客户端")
        return _api_client_search
    else:
        # 默认客户端（向后兼容）
        if _api_client is None:
            _api_client = YouTubeAPIClient(use_for=None)
        return _api_client


def yt_get(
    endpoint: str, 
    params: Dict[str, Any], 
    max_retries: int | None = None,
    use_for: str | None = None
) -> Dict[str, Any]:
    """
    便捷函数：调用 YouTube API
    
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
    client = get_api_client(use_for=use_for)
    return client.get(endpoint, params, max_retries)

