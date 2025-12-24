"""
候选频道收集模块
通过各种方式收集可能相似的频道候选
"""
from typing import List

from infrastructure.config import Config
from infrastructure.logger import get_logger
from core.youtube_api import YouTubeAPIError, YouTubeQuotaExceededError, yt_get

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
    
    # 失败计数器：如果连续失败太多次，提前停止以避免浪费配额
    consecutive_failures = 0
    # 从配置读取，如果没有配置则使用默认值2（更早停止以节省配额）
    max_consecutive_failures = Config.CANDIDATE_COLLECTION.get("max_consecutive_failures", 2)

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
            # 成功时重置失败计数器
            consecutive_failures = 0
        except YouTubeQuotaExceededError:
            logger.error(f"搜索相关视频时配额已用完: {vid}")
            raise
        except YouTubeAPIError as e:
            consecutive_failures += 1
            error_msg = str(e)
            # 如果是"invalid argument"错误，可能是视频ID本身有问题，记录但不继续
            if "invalid argument" in error_msg.lower():
                logger.warning(f"视频ID无效（可能已删除或私有），跳过: {vid}")
            else:
                logger.warning(f"YouTube API 错误，搜索相关视频失败: {vid}, 错误: {e}")
            
            # 如果连续失败太多次，提前停止以避免浪费配额（阈值已从3减少到2）
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(
                    f"相关视频搜索连续失败 {consecutive_failures} 次，"
                    f"提前停止以避免浪费配额。已收集 {len(candidate_channel_ids)} 个候选频道。"
                )
                break
            continue
        except Exception as e:
            consecutive_failures += 1
            logger.warning(f"搜索相关视频时发生错误: {vid}, 错误: {e}", exc_info=True)
            
            # 如果连续失败太多次，提前停止
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(
                    f"相关视频搜索连续失败 {consecutive_failures} 次，"
                    f"提前停止以避免浪费配额。已收集 {len(candidate_channel_ids)} 个候选频道。"
                )
                break
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

