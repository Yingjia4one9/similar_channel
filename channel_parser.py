"""
频道 URL 解析模块
处理各种 YouTube 频道/视频链接格式，提取频道 ID
"""
import re
from urllib.parse import parse_qs, urlparse

from logger import get_logger
from youtube_api import yt_get

logger = get_logger()

# 允许的YouTube域名（防止SSRF攻击）
ALLOWED_DOMAINS = {
    "www.youtube.com",
    "youtube.com",
    "m.youtube.com",
    "youtu.be",
}


def get_channel_id_by_handle(handle: str, use_for: str | None = None) -> str:
    """
    通过新样式 handle（@xxx）获取频道 ID。
    
    Args:
        handle: 频道 handle（如 "Crypto621"）
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID
    
    Raises:
        ValueError: 如果找不到对应的频道
    """
    handle = handle.lstrip("@")
    data = yt_get(
        "channels",
        {
            "part": "id",
            "forHandle": handle,
            "fields": "items(id)",
        },
        use_for=use_for,
    )
    items = data.get("items", [])
    if not items:
        raise ValueError(f"未找到 handle 为 @{handle} 的频道。")
    return items[0]["id"]


def get_channel_id_by_username(username: str, use_for: str | None = None) -> str:
    """
    通过 legacy username（/user/xxx）获取频道 ID。
    
    Args:
        username: 频道 username
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID
    
    Raises:
        ValueError: 如果找不到对应的频道
    """
    data = yt_get(
        "channels",
        {
            "part": "id",
            "forUsername": username,
            "fields": "items(id)",
        },
        use_for=use_for,
    )
    items = data.get("items", [])
    if not items:
        raise ValueError(f"未找到 username 为 {username} 的频道。")
    return items[0]["id"]


def get_channel_id_by_custom_url_segment(segment: str, use_for: str | None = None) -> str:
    """
    对于 /c/xxx 或根路径 /xxx，YouTube API 没有直接参数，
    这里通过搜索频道名称的方式近似匹配。
    
    Args:
        segment: 自定义 URL 段
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID
    
    Raises:
        ValueError: 如果找不到对应的频道
    """
    data = yt_get(
        "search",
        {
            "part": "snippet",
            "q": segment,
            "type": "channel",
            "maxResults": 5,
            "fields": "items(snippet/channelId)",
        },
        use_for=use_for,
    )
    items = data.get("items", [])
    if not items:
        raise ValueError(f"未通过自定义路径 {segment} 找到频道。")
    # 取搜索结果的第一个频道
    return items[0]["snippet"]["channelId"]


def get_channel_id_by_video_id(video_id: str, use_for: str | None = None) -> str:
    """
    通过视频 ID 获取频道 ID。
    
    Args:
        video_id: 视频 ID
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID
    
    Raises:
        ValueError: 如果找不到对应的频道
    """
    data = yt_get(
        "videos",
        {
            "part": "snippet",
            "id": video_id,
            "fields": "items(snippet/channelId)",
        },
        use_for=use_for,
    )
    items = data.get("items", [])
    if not items:
        raise ValueError("无法通过该视频 ID 找到对应频道。")
    return items[0]["snippet"]["channelId"]


def extract_channel_id_from_url(url: str, use_for: str | None = None) -> str:
    """
    支持多种常见频道 / 视频链接形式：
    - https://www.youtube.com/channel/CHANNEL_ID
    - https://youtube.com/@handle
    - https://youtube.com/user/USERNAME
    - https://youtube.com/c/CUSTOM_NAME
    - https://www.youtube.com/某名字（根路径，尝试按自定义路径匹配）
    - https://www.youtube.com/watch?v=VIDEO_ID...
    - https://youtu.be/VIDEO_ID
    
    Args:
        url: YouTube 频道或视频链接
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID
    
    Raises:
        ValueError: 如果无法从链接解析出频道 ID 或 URL 不安全
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL不能为空且必须是字符串")
    
    # 清理URL（去除前后空白）
    url = url.strip()
    
    if not url:
        raise ValueError("URL不能为空")
    
    # 验证URL长度（防止过长的URL导致DoS）
    if len(url) > 2048:
        raise ValueError("URL长度超过限制（最大2048字符）")
    
    # 解析URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.warning(f"URL解析失败: {url[:50]}... (错误: {e})")
        raise ValueError(f"无效的URL格式: {e}") from e
    
    # 验证域名（防止SSRF攻击）
    domain = parsed.netloc.lower()
    if not domain:
        raise ValueError("URL必须包含域名")
    
    # 移除端口号（如果有）
    if ":" in domain:
        domain = domain.split(":")[0]
    
    # 检查是否为允许的YouTube域名
    if domain not in ALLOWED_DOMAINS:
        logger.warning(f"拒绝非YouTube域名: {domain}")
        raise ValueError(
            f"只支持YouTube域名（youtube.com, youtu.be等），"
            f"检测到: {domain}"
        )
    
    # 验证协议（只允许http和https）
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"只支持HTTP/HTTPS协议，检测到: {scheme}")
    
    path = parsed.path or ""

    # youtu.be 短链接，直接当作视频链接处理
    if parsed.netloc in {"youtu.be"} and path:
        video_id = path.lstrip("/")
        if video_id:
            return get_channel_id_by_video_id(video_id, use_for=use_for)

    # /channel/CHANNEL_ID
    m = re.match(r"^/channel/([a-zA-Z0-9_-]+)", path)
    if m:
        return m.group(1)

    # handle: /@xxx
    m = re.match(r"^/@([a-zA-Z0-9._-]+)", path)
    if m:
        handle = m.group(1)
        return get_channel_id_by_handle(handle, use_for=use_for)

    # legacy username: /user/xxx
    m = re.match(r"^/user/([a-zA-Z0-9._-]+)", path)
    if m:
        username = m.group(1)
        return get_channel_id_by_username(username, use_for=use_for)

    # custom url: /c/xxx
    m = re.match(r"^/c/([^/]+)", path)
    if m:
        segment = m.group(1)
        return get_channel_id_by_custom_url_segment(segment, use_for=use_for)

    # watch 链接，解析 v=VIDEO_ID，再通过 video 查 channelId
    if "watch" in path:
        qs = parse_qs(parsed.query or "")
        video_id_list = qs.get("v")
        if video_id_list:
            video_id = video_id_list[0]
            return get_channel_id_by_video_id(video_id, use_for=use_for)

    # 根路径 /xxx 形式，尝试按自定义路径匹配
    root_segment = path.strip("/")
    if root_segment:
        return get_channel_id_by_custom_url_segment(root_segment, use_for=use_for)

    raise ValueError("暂时无法从该链接解析出频道 ID，请提供频道主页或任意视频链接。")

