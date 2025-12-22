"""
频道信息获取模块
处理频道基本信息、视频列表、统计数据等
"""
import time
from typing import Any, Dict, List

from cache import cached_channel_info, cached_channel_videos
from config import Config
from logger import get_logger
from youtube_api import YouTubeAPIError, YouTubeQuotaExceededError, yt_get

logger = get_logger()

# 批量请求统计（CP-y3-09：批量请求优化）
_batch_request_stats = {
    "total_batch_requests": 0,  # 总批量请求次数
    "total_single_requests": 0,  # 总单请求次数（回退）
    "total_channels_batched": 0,  # 通过批量请求获取的频道数
    "total_channels_single": 0,  # 通过单请求获取的频道数（回退）
    "batch_failures": 0,  # 批量请求失败次数
    "total_time_saved": 0.0,  # 节省的总时间（秒，估算）
}


@cached_channel_info(ttl=7200)  # 缓存2小时
def get_channel_basic_info(channel_id: str, use_for: str | None = None) -> Dict[str, Any]:
    """
    获取频道基础信息。
    
    Args:
        channel_id: 频道 ID
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        包含频道信息的字典
    
    Raises:
        ValueError: 如果找不到该频道
    """
    try:
        data = yt_get(
            "channels",
            {
                "part": "snippet,statistics",
                "id": channel_id,
                "fields": "items(id,snippet(title,description,thumbnails,defaultLanguage,country),statistics(subscriberCount,videoCount,viewCount))",
            },
            use_for=use_for,
        )
    except YouTubeQuotaExceededError:
        logger.error(f"获取频道信息时配额已用完: {channel_id}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API 错误，获取频道信息失败: {channel_id}, 错误: {e}")
        raise ValueError(f"无法获取频道信息: {e}") from e
    except Exception as e:
        logger.error(f"获取频道信息时发生未知错误: {channel_id}, 错误: {e}", exc_info=True)
        raise ValueError(f"获取频道信息失败: {e}") from e
    
    items = data.get("items", []) if data else []
    if not items or len(items) == 0:
        raise ValueError("未找到该频道，channelId 可能无效。")
    
    item = items[0]
    if not item or not isinstance(item, dict):
        raise ValueError("频道数据格式无效")
    
    snippet = item.get("snippet") if item.get("snippet") else {}
    statistics = item.get("statistics") if item.get("statistics") else {}
    
    # 安全获取channelId
    channel_id = item.get("id")
    if not channel_id:
        raise ValueError("频道ID缺失")
    
    return {
        "channelId": channel_id,
        "title": snippet.get("title") or "" if snippet else "",
        "description": snippet.get("description") or "" if snippet else "",
        "thumbnails": snippet.get("thumbnails") or {} if snippet else {},
        "subscriberCount": int(statistics.get("subscriberCount") or 0) if statistics else 0,
        "videoCount": int(statistics.get("videoCount") or 0) if statistics else 0,
        "viewCount": int(statistics.get("viewCount") or 0) if statistics else 0,
        # 语言 / 地区信息（可能为 None）
        "defaultLanguage": snippet.get("defaultLanguage") if snippet else None,
        "defaultAudioLanguage": snippet.get("defaultAudioLanguage") if snippet else None,
        "country": snippet.get("country") if snippet else None,
    }


def get_recent_video_ids(channel_id: str, max_results: int | None = None, use_for: str | None = None) -> List[str]:
    """
    获取该频道最近发布的一些视频 ID。
    
    Args:
        channel_id: 频道 ID
        max_results: 最大返回数量（默认使用配置值）
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        视频 ID 列表
    """
    max_results = max_results or Config.CHANNEL_INFO["recent_videos_count"]
    try:
        data = yt_get(
            "search",
            {
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": max_results,
                "type": "video",
                "fields": "items(id/videoId)",
            },
            use_for=use_for,
        )
    except YouTubeQuotaExceededError:
        logger.error(f"获取视频ID列表时配额已用完: {channel_id}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API 错误，获取视频ID列表失败: {channel_id}, 错误: {e}")
        raise ValueError(f"无法获取视频ID列表: {e}") from e
    except Exception as e:
        logger.error(f"获取视频ID列表时发生未知错误: {channel_id}, 错误: {e}", exc_info=True)
        raise ValueError(f"获取视频ID列表失败: {e}") from e
    video_ids: List[str] = []
    items = data.get("items", []) if data else []
    for item in items:
        if not item or not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not item_id or not isinstance(item_id, dict):
            continue
        vid = item_id.get("videoId")
        if vid and isinstance(vid, str):
            video_ids.append(vid)
    return video_ids


@cached_channel_videos(ttl=1800)  # 缓存30分钟
def get_recent_video_snippets_for_channel(
    channel_id: str, max_results: int | None = None, use_for: str | None = None
) -> List[Dict[str, Any]]:
    """
    为相似度计算准备：获取频道最近的一些视频标题与简介（只用 search 的 snippet，就不再额外查 videos）。
    同时返回视频ID和缩略图信息，用于前端显示。
    
    Args:
        channel_id: 频道 ID
        max_results: 最大返回数量（默认使用配置值）
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        视频信息列表
    """
    max_results = max_results or Config.CHANNEL_INFO["recent_videos_count"]
    try:
        data = yt_get(
            "search",
            {
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": max_results,
                "type": "video",
                "fields": "items(id/videoId,snippet(title,description,thumbnails))",
            },
            use_for=use_for,
        )
    except YouTubeQuotaExceededError:
        logger.error(f"获取视频列表时配额已用完: {channel_id}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API 错误，获取视频列表失败: {channel_id}, 错误: {e}")
        raise ValueError(f"无法获取视频列表: {e}") from e
    except Exception as e:
        logger.error(f"获取视频列表时发生未知错误: {channel_id}, 错误: {e}", exc_info=True)
        raise ValueError(f"获取视频列表失败: {e}") from e
    videos: List[Dict[str, Any]] = []
    items = data.get("items", []) if data else []
    for item in items:
        if not item or not isinstance(item, dict):
            continue
        
        snip = item.get("snippet") if item.get("snippet") else {}
        item_id = item.get("id")
        video_id = ""
        if item_id and isinstance(item_id, dict):
            video_id = item_id.get("videoId") or ""
        elif isinstance(item_id, str):
            video_id = item_id
        
        if not video_id:
            continue
        
        thumbnails = snip.get("thumbnails") if snip and snip.get("thumbnails") else {}
        videos.append(
            {
                "videoId": video_id,
                "title": snip.get("title", "") or "",
                "description": snip.get("description", "") or "",
                "thumbnails": thumbnails,
            }
        )
    return videos


def get_recent_videos_stats(channel_id: str, max_results: int | None = None, use_for: str | None = None) -> Dict[str, float]:
    """
    获取频道最近视频的统计数据（观看数、点赞数、评论数），用于计算 E.R. 和 V.R.。
    返回平均观看数、平均点赞数、平均评论数。
    
    根据 SimilarTube 的计算方式：
    - E.R. (Engagement Rate) = 平均点赞数 / 订阅数 * 100%
    - V.R. (View Rate) = 平均观看数 / 订阅数 * 100%
    
    Args:
        channel_id: 频道 ID
        max_results: 最大统计视频数量（默认使用配置值）
    
    Returns:
        包含平均观看数、平均点赞数、平均评论数的字典
    """
    max_results = max_results or Config.CHANNEL_INFO["stats_videos_count"]
    # 先获取最近视频的 ID
    video_ids = get_recent_video_ids(channel_id, max_results=max_results, use_for=use_for)
    if not video_ids:
        return {"avg_views": 0.0, "avg_likes": 0.0, "avg_comments": 0.0}
    
    # 批量获取视频统计数据（YouTube API 一次最多 50 个）
    all_views: List[int] = []
    all_likes: List[int] = []
    all_comments: List[int] = []
    
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            data = yt_get(
                "videos",
                {
                    "part": "statistics",
                    "id": ",".join(batch),
                    "fields": "items(statistics(viewCount,likeCount,commentCount))",
                },
                use_for=use_for,
            )
        except YouTubeQuotaExceededError:
            logger.error(f"获取视频统计数据时配额已用完")
            raise
        except YouTubeAPIError as e:
            logger.warning(f"YouTube API 错误，获取视频统计数据失败: {e}")
            # 继续处理其他批次，不中断整个流程
            continue
        except Exception as e:
            logger.warning(f"获取视频统计数据时发生错误: {e}", exc_info=True)
            # 继续处理其他批次，不中断整个流程
            continue
        for item in data.get("items", []):
            stats = item.get("statistics", {}) or {}
            view_count = int(stats.get("viewCount", 0) or 0)
            like_count = int(stats.get("likeCount", 0) or 0)
            comment_count = int(stats.get("commentCount", 0) or 0)
            
            if view_count > 0:  # 只统计有观看数的视频
                all_views.append(view_count)
                all_likes.append(like_count)
                all_comments.append(comment_count)
    
    if not all_views:
        return {"avg_views": 0.0, "avg_likes": 0.0, "avg_comments": 0.0}
    
    return {
        "avg_views": float(sum(all_views) / len(all_views)),
        "avg_likes": float(sum(all_likes) / len(all_likes)),
        "avg_comments": float(sum(all_comments) / len(all_comments)),
    }


def batch_get_channels_info(channel_ids: List[str], use_for: str | None = None) -> List[Dict[str, Any]]:
    """
    按照 YouTube API 限制，一次最多查 50 个频道，并在批量失败时回退到单请求。
    支持配置化批量大小，并记录批量请求统计（CP-y3-09：批量请求优化）。
    
    Args:
        channel_ids: 频道 ID 列表
    
    Returns:
        频道信息列表
    """
    results: List[Dict[str, Any]] = []
    if not channel_ids:
        return results
    
    # 使用配置的批量大小（CP-y3-09：批量请求优化）
    batch_size = Config.CHANNEL_INFO.get("batch_size", 50)
    fallback_singles = 0
    total = len(channel_ids)
    start_time = time.time()
    logger.debug(f"批量获取频道信息：共 {total} 个，批大小 {batch_size}")
    
    # 更新统计：总批量请求次数
    global _batch_request_stats
    _batch_request_stats["total_batch_requests"] += ((total - 1) // batch_size) + 1
    
    for i in range(0, total, batch_size):
        batch = channel_ids[i : i + batch_size]
        batch_results: List[Dict[str, Any]] = []
        batch_start_time = time.time()
        try:
            data = yt_get(
                "channels",
                {
                    "part": "snippet,statistics",
                    "id": ",".join(batch),
                    "fields": "items(id,snippet(title,description,thumbnails,defaultLanguage,country),statistics(subscriberCount,videoCount,viewCount))",
                },
                use_for=use_for,
            )
            # 批量请求成功：更新统计（CP-y3-09：批量请求优化）
            batch_time = time.time() - batch_start_time
            _batch_request_stats["total_channels_batched"] += len(batch)
            # 估算节省的时间：假设单个请求平均0.5秒，批量请求节省 (len(batch) - 1) * 0.5 秒
            estimated_time_saved = (len(batch) - 1) * 0.5
            _batch_request_stats["total_time_saved"] += estimated_time_saved
        except YouTubeQuotaExceededError:
            logger.error("批量获取频道信息时配额已用完")
            _batch_request_stats["batch_failures"] += 1
            raise
        except YouTubeAPIError as e:
            logger.warning(f"批量获取频道信息失败，回退到单请求: {e}")
            _batch_request_stats["batch_failures"] += 1
            _batch_request_stats["total_single_requests"] += len(batch)
            for cid in batch:
                try:
                    batch_results.append(get_channel_basic_info(cid))
                    fallback_singles += 1
                    _batch_request_stats["total_channels_single"] += 1
                except Exception as single_err:
                    logger.debug(f"单个频道获取失败（批量回退）: {cid}, 错误: {single_err}")
            results.extend(batch_results)
            continue
        except Exception as e:
            logger.warning(f"批量获取频道信息时发生错误，回退到单请求: {e}", exc_info=True)
            _batch_request_stats["batch_failures"] += 1
            _batch_request_stats["total_single_requests"] += len(batch)
            for cid in batch:
                try:
                    batch_results.append(get_channel_basic_info(cid))
                    fallback_singles += 1
                    _batch_request_stats["total_channels_single"] += 1
                except Exception as single_err:
                    logger.debug(f"单个频道获取失败（批量回退）: {cid}, 错误: {single_err}")
            results.extend(batch_results)
            continue
        
        items = data.get("items", []) if data else []
        returned_ids = set()
        for item in items:
            snippet = item.get("snippet", {}) or {}
            statistics = item.get("statistics", {}) or {}
            cid = item.get("id")
            if cid:
                returned_ids.add(cid)
            batch_results.append(
                {
                    "channelId": cid,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "thumbnails": snippet.get("thumbnails", {}),
                    "subscriberCount": int(statistics.get("subscriberCount", 0) or 0),
                    "videoCount": int(statistics.get("videoCount", 0) or 0),
                    "viewCount": int(statistics.get("viewCount", 0) or 0),
                    "defaultLanguage": snippet.get("defaultLanguage"),
                    "defaultAudioLanguage": snippet.get("defaultAudioLanguage"),
                    "country": snippet.get("country"),
                }
            )
        
        # 如果批量返回中有缺失的频道，回退到单请求补齐
        missing_ids = [cid for cid in batch if cid not in returned_ids]
        if missing_ids:
            _batch_request_stats["total_single_requests"] += len(missing_ids)
        for cid in missing_ids:
            try:
                batch_results.append(get_channel_basic_info(cid))
                fallback_singles += 1
                _batch_request_stats["total_channels_single"] += 1
            except Exception as single_err:
                logger.debug(f"单个频道获取失败（批量缺失回补）: {cid}, 错误: {single_err}")
        
        results.extend(batch_results)
    
    elapsed = time.time() - start_time
    batch_count = ((total - 1) // batch_size) + 1
    logger.info(
        f"批量获取频道信息完成：总数 {total}，批量分组 {batch_count}，"
        f"回退单请求 {fallback_singles}，耗时 {elapsed:.2f} 秒"
    )
    return results


def get_batch_request_stats() -> Dict[str, Any]:
    """
    获取批量请求统计信息（CP-y3-09：批量请求优化）
    
    Returns:
        包含批量请求统计的字典
    """
    global _batch_request_stats
    total_channels = _batch_request_stats["total_channels_batched"] + _batch_request_stats["total_channels_single"]
    batch_rate = (
        (_batch_request_stats["total_channels_batched"] / total_channels * 100)
        if total_channels > 0
        else 0.0
    )
    
    return {
        "total_batch_requests": _batch_request_stats["total_batch_requests"],
        "total_single_requests": _batch_request_stats["total_single_requests"],
        "total_channels_batched": _batch_request_stats["total_channels_batched"],
        "total_channels_single": _batch_request_stats["total_channels_single"],
        "batch_failures": _batch_request_stats["batch_failures"],
        "total_time_saved_seconds": round(_batch_request_stats["total_time_saved"], 2),
        "batch_rate_percent": round(batch_rate, 2),  # 批量请求占比
        "estimated_quota_saved": _batch_request_stats["total_batch_requests"],  # 估算节省的配额（批量请求次数）
    }

