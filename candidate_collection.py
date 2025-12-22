"""
候选频道收集模块
通过各种方式收集可能相似的频道候选
"""
from typing import List

from config import Config
from logger import get_logger
from youtube_api import YouTubeAPIError, YouTubeQuotaExceededError, yt_get

logger = get_logger()


def collect_candidate_channels_from_related_videos(
    video_ids: List[str], per_video: int | None = None, limit: int | None = None, use_for: str | None = None
) -> List[str]:
    """
    对每个视频调用一次 relatedToVideoId 搜索，收集可能相似的频道。
    
    Args:
        video_ids: 视频 ID 列表
        per_video: 每个视频搜索的频道数量（默认使用配置值）
        limit: 总限制（默认使用配置值）
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        候选频道 ID 列表
    """
    per_video = per_video or Config.CANDIDATE_COLLECTION["related_videos_per_video"]
    limit = limit or Config.CANDIDATE_COLLECTION["related_videos_limit"]
    
    candidate_channel_ids: List[str] = []
    seen = set()

    for vid in video_ids:
        if len(candidate_channel_ids) >= limit:
            break
        try:
            data = yt_get(
                "search",
                {
                    "part": "snippet",
                    "relatedToVideoId": vid,
                    "type": "video",
                    "maxResults": per_video,
                    "fields": "items(snippet/channelId)",
                },
                use_for=use_for,
            )
        except YouTubeQuotaExceededError:
            logger.error(f"搜索相关视频时配额已用完: {vid}")
            raise
        except YouTubeAPIError as e:
            logger.warning(f"YouTube API 错误，搜索相关视频失败: {vid}, 错误: {e}")
            # 单个视频失败不影响其他视频
            continue
        except Exception as e:
            logger.warning(f"搜索相关视频时发生错误: {vid}, 错误: {e}", exc_info=True)
            # 单个视频失败不影响其他视频
            continue
        for item in data.get("items", []):
            ch_id = item.get("snippet", {}).get("channelId")
            if ch_id and ch_id not in seen:
                seen.add(ch_id)
                candidate_channel_ids.append(ch_id)
                if len(candidate_channel_ids) >= limit:
                    break

    return candidate_channel_ids


def search_candidate_channels_by_title(title: str, limit: int | None = None, use_for: str | None = None) -> List[str]:
    """
    备用方案：根据频道标题做一次全局搜索，收集频道候选。
    当基于 relatedToVideoId 的方式因为某些特殊视频而报错时，用这个方式兜底。
    
    Args:
        title: 搜索关键词
        limit: 最大返回数量（默认使用配置值）
        use_for: API key 用途标识，可选值："index"（索引构建）、"search"（实时搜索）
    
    Returns:
        频道 ID 列表
    """
    limit = limit or Config.CANDIDATE_COLLECTION["title_search_limit"]
    try:
        data = yt_get(
            "search",
            {
                "part": "snippet",
                "q": title,
                "type": "channel",
                "maxResults": min(limit, 50),
                    "fields": "items(snippet/channelId)",
                },
                use_for=use_for,
        )
    except YouTubeQuotaExceededError:
        logger.error(f"按标题搜索频道时配额已用完: {title}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API 错误，按标题搜索频道失败: {title}, 错误: {e}")
        raise ValueError(f"搜索频道失败: {e}") from e
    except Exception as e:
        logger.error(f"按标题搜索频道时发生未知错误: {title}, 错误: {e}", exc_info=True)
        raise ValueError(f"搜索频道失败: {e}") from e
    ids: List[str] = []
    seen = set()
    for item in data.get("items", []):
        ch_id = item.get("snippet", {}).get("channelId")
        if ch_id and ch_id not in seen:
            seen.add(ch_id)
            ids.append(ch_id)
    return ids

