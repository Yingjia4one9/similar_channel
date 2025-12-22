import asyncio
import json
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from logger import get_logger
from config import Config
from quota_tracker import (
    get_quota_usage_today,
    get_quota_usage_by_endpoint,
    get_quota_status,
    is_quota_exceeded,
    reset_quota_stats,
    get_fallback_stats,
    get_quota_statistics,
    get_quota_usage_rate,
    check_and_record_quota_warning,
    get_quota_warnings,
    acknowledge_quota_warning,
    get_rate_limit_status,
    predict_quota_exhaustion,
    get_quota_usage_logs,
    clean_legacy_quota_records,
)
from result_cache import (
    generate_result_id,
    store_result,
    get_result,
    get_result_by_params,
    get_cache_stats,
    generate_cache_key,
)
from xss_protection import validate_and_sanitize_request
from youtube_client import get_similar_channels_by_url
from channel_info import get_batch_request_stats

logger = get_logger()


app = FastAPI(title="YouTube Similar Channel Backend")

# CORS配置：允许的来源可通过环境变量配置
# 开发环境：默认允许所有来源（支持本地file://打开）
# 生产环境：应设置CORS_ALLOW_ORIGINS环境变量，如 "https://example.com,https://www.example.com"
cors_origins = Config.get_cors_allow_origins()
logger.info(f"CORS允许的来源: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# XSS防护：添加安全响应头（CP-y5-07）
@app.middleware("http")
async def add_security_headers(request, call_next):
    """
    为所有响应添加安全头，防止XSS攻击
    """
    response = await call_next(request)
    
    # Content-Security-Policy：限制资源加载来源
    # 允许内联脚本和样式（前端HTML需要），但限制外部资源
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "  # 允许内联脚本（前端HTML需要）
        "style-src 'self' 'unsafe-inline'; "    # 允许内联样式（前端HTML需要）
        "img-src 'self' data: https:; "         # 允许图片（YouTube缩略图）
        "font-src 'self' data:; "
        "connect-src 'self' http://127.0.0.1:8000 http://localhost:8000; "  # 允许API连接
        "frame-ancestors 'none'; "              # 防止点击劫持
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # X-Content-Type-Options：防止MIME类型嗅探
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # X-Frame-Options：防止点击劫持（与CSP的frame-ancestors重复，但提供兼容性）
    response.headers["X-Frame-Options"] = "DENY"
    
    # X-XSS-Protection：启用浏览器XSS过滤器（已废弃，但提供兼容性）
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Referrer-Policy：控制referrer信息
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


class SimilarChannelRequest(BaseModel):
    """相似频道搜索请求模型"""
    channel_url: Annotated[str, Field(..., description="YouTube 频道链接")]
    max_results: Annotated[int, Field(default=20, ge=1, le=200, description="最大返回结果数（1-200）")]
    min_subscribers: Annotated[int | None, Field(default=None, ge=0, description="最小订阅数筛选")]
    max_subscribers: Annotated[int | None, Field(default=None, ge=0, description="最大订阅数筛选")]
    preferred_language: str | None = Field(default=None, description="偏好语言（如 'en', 'zh-Hans'）")
    preferred_region: str | None = Field(default=None, description="偏好地区（如 'US', 'CN'）")
    min_similarity: Annotated[float | None, Field(default=None, ge=0.0, le=1.0, description="最低语义相似度（0-1）")]
    bd_mode: bool = Field(default=False, description="BD模式：交易所BD寻找KOL专用，启用后会计算合约聚焦度、商业化潜力等指标")
    
    @field_validator('channel_url')
    @classmethod
    def validate_channel_url(cls, v: str) -> str:
        """验证频道链接不为空"""
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("频道链接不能为空")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_subscriber_range(self):
        """验证订阅数范围"""
        if self.min_subscribers is not None and self.max_subscribers is not None:
            if self.min_subscribers > self.max_subscribers:
                raise ValueError("min_subscribers 不能大于 max_subscribers")
        return self


class SimilarChannelResponse(BaseModel):
    """相似频道搜索响应模型"""
    result_id: str = Field(description="结果 ID，可用于后续导出 CSV")
    base_channel: dict
    similar_channels: list[dict]
    quota_exceeded_channels: list[dict] = Field(default_factory=list, description="因配额不足而无法获取完整信息的频道列表（包含channelId和channelUrl）")
    failed_channels: list[dict] = Field(default_factory=list, description="因其他原因失败的频道列表（包含channelId、channelUrl和reason）")
    bd_summary: dict | None = Field(default=None, description="BD模式统计摘要（仅在bd_mode=True时返回）")


async def execute_search(payload: SimilarChannelRequest) -> dict:
    """
    执行相似频道搜索（提取重复逻辑）。
    参数验证已由 Pydantic 自动处理。
    """
    try:
        # XSS防护：验证和清理请求参数（CP-y5-07）
        payload_dict = payload.model_dump()
        sanitized_payload = validate_and_sanitize_request(payload_dict)
        
        # 构建请求参数字典（用于缓存键生成）
        request_params = {
            "channel_url": sanitized_payload.get("channel_url", payload.channel_url),
            "max_results": sanitized_payload.get("max_results", payload.max_results),
            "min_subscribers": sanitized_payload.get("min_subscribers", payload.min_subscribers),
            "max_subscribers": sanitized_payload.get("max_subscribers", payload.max_subscribers),
            "preferred_language": sanitized_payload.get("preferred_language", payload.preferred_language),
            "preferred_region": sanitized_payload.get("preferred_region", payload.preferred_region),
            "min_similarity": sanitized_payload.get("min_similarity", payload.min_similarity),
            "bd_mode": sanitized_payload.get("bd_mode", payload.bd_mode),
        }
        
        # 尝试从缓存获取结果（基于请求参数）
        cached_result = get_result_by_params(request_params)
        if cached_result:
            logger.info("从缓存返回搜索结果")
            return cached_result
        
        # 缓存未命中，执行搜索
        result = await get_similar_channels_by_url(
            channel_url=payload.channel_url,
            max_results=payload.max_results,
            min_subscribers=payload.min_subscribers,
            max_subscribers=payload.max_subscribers,
            preferred_language=payload.preferred_language,
            preferred_region=payload.preferred_region,
            min_similarity=payload.min_similarity,
            bd_mode=payload.bd_mode,
        )
        
        # 生成结果 ID 并缓存结果（包含缓存键）
        result_id = generate_result_id()
        cache_key = generate_cache_key(request_params)
        store_result(result_id, result, cache_key=cache_key)
        result["result_id"] = result_id
        
        return result
    except ValueError as e:
        logger.warning(f"请求参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 错误消息不泄露敏感信息（CP-y5-13：错误信息安全）
        logger.error(f"处理请求时发生错误: {type(e).__name__}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="处理请求时发生内部错误，请稍后重试或联系管理员。"
        )


@app.post("/similar-channels", response_model=SimilarChannelResponse)
async def similar_channels(payload: SimilarChannelRequest):
    """
    输入一个 YouTube 频道链接，返回若干相似频道及相似度。
    """
    return await execute_search(payload)


class ExportRequest(BaseModel):
    """导出请求模型"""
    result_id: str | None = Field(default=None, description="结果 ID，如果提供则直接使用缓存的结果")
    # 如果 result_id 为 None，则使用以下参数进行搜索
    channel_url: str | None = Field(default=None, description="YouTube 频道链接（仅在 result_id 为空时使用）")
    max_results: int | None = Field(default=None, ge=1, le=200, description="最大返回结果数（仅在 result_id 为空时使用）")
    min_subscribers: int | None = Field(default=None, ge=0, description="最小订阅数筛选（仅在 result_id 为空时使用）")
    max_subscribers: int | None = Field(default=None, ge=0, description="最大订阅数筛选（仅在 result_id 为空时使用）")
    preferred_language: str | None = Field(default=None, description="偏好语言（仅在 result_id 为空时使用）")
    preferred_region: str | None = Field(default=None, description="偏好地区（仅在 result_id 为空时使用）")
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0, description="最低语义相似度（仅在 result_id 为空时使用）")


@app.post("/similar-channels/export", response_class=PlainTextResponse)
async def similar_channels_export(payload: ExportRequest):
    """
    导出搜索结果为 CSV 格式。
    
    优先使用 result_id 从缓存获取结果（不消耗 API 配额）。
    如果 result_id 为空，则使用其他参数进行搜索（会消耗 API 配额）。
    """
    # 优先使用缓存的结果
    if payload.result_id:
        results = get_result(payload.result_id)
        if results is None:
            raise HTTPException(
                status_code=404,
                detail=f"结果 ID '{payload.result_id}' 不存在或已过期。请重新搜索或使用有效的 result_id。"
            )
    else:
        # 如果没有提供 result_id，则使用传统方式搜索（向后兼容）
        if not payload.channel_url:
            raise HTTPException(
                status_code=400,
                detail="必须提供 result_id 或 channel_url"
            )
        
        search_payload = SimilarChannelRequest(
            channel_url=payload.channel_url,
            max_results=payload.max_results or 20,
            min_subscribers=payload.min_subscribers,
            max_subscribers=payload.max_subscribers,
            preferred_language=payload.preferred_language,
            preferred_region=payload.preferred_region,
            min_similarity=payload.min_similarity,
        )
        results = await execute_search(search_payload)

    base = results["base_channel"]
    similars = results["similar_channels"]
    is_bd_mode = results.get("bd_summary") is not None

    # 构造 CSV：UTF-8 带 BOM，便于在 Excel 中正常显示中文
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    
    # 根据是否为BD模式选择不同的CSV表头
    if is_bd_mode:
        writer.writerow(
            [
                "base_channel_title",
                "base_channel_url",
                "channel_title",
                "channel_url",
                "subscriberCount",
                "viewCount",
                "engagement_rate",
                "bd_priority",
                "bd_total_score",
                "contract_focus_score",
                "audience_quality_score",
                "commercialization_score",
                "has_competitor_collab",
                "competitors",
                "has_email",
                "emails",
                "topics",
                "audience",
                "recommendation_reasons",
                "concerns",
            ]
        )
    else:
        writer.writerow(
            [
                "base_channel_title",
                "base_channel_url",
                "channel_title",
                "channel_url",
                "subscriberCount",
                "viewCount",
                "similarity",
                "scale_score",
                "total_score",
                "emails",
            ]
        )

    base_url = f"https://www.youtube.com/channel/{base.get('channelId')}"

    for ch in similars:
        ch_url = f"https://www.youtube.com/channel/{ch.get('channelId')}"
        emails = ch.get("emails") or []
        
        if is_bd_mode:
            # BD模式：导出BD相关字段
            bd_metrics = ch.get("bd_metrics", {})
            bd_breakdown = ch.get("bd_breakdown", {})
            bd_recommendation = ch.get("bd_recommendation", {})
            competitor_detection = bd_metrics.get("competitor_detection", {})
            commercialization = bd_metrics.get("commercialization", {})
            
            writer.writerow(
                [
                    base.get("title", ""),
                    base_url,
                    ch.get("title", ""),
                    ch_url,
                    ch.get("subscriberCount", 0),
                    ch.get("viewCount", 0),
                    round(ch.get("engagement_rate", 0.0), 2),
                    ch.get("bd_priority", ""),
                    round(ch.get("bd_total_score", 0.0), 4),
                    round(bd_breakdown.get("contract_focus_score", 0.0), 4),
                    round(bd_breakdown.get("audience_quality_score", 0.0), 4),
                    round(bd_breakdown.get("commercialization_score", 0.0), 4),
                    "是" if competitor_detection.get("has_competitor_collab", False) else "否",
                    ";".join(competitor_detection.get("competitors", [])),
                    "是" if commercialization.get("has_email", False) else "否",
                    ";".join(emails),
                    ";".join(ch.get("topics", [])),
                    ";".join(ch.get("audience", [])),
                    ";".join(bd_recommendation.get("reasons", [])),
                    ";".join(bd_recommendation.get("concerns", [])),
                ]
            )
        else:
            writer.writerow(
                [
                    base.get("title", ""),
                    base_url,
                    ch.get("title", ""),
                    ch_url,
                    ch.get("subscriberCount", 0),
                    ch.get("viewCount", 0),
                    round(ch.get("similarity", 0.0), 4),
                    round(ch.get("scale_score", 0.0), 4),
                    round(ch.get("total_score", 0.0), 4),
                    ";".join(emails),
                ]
            )

    csv_text = output.getvalue()
    # 添加 BOM
    csv_text = "\ufeff" + csv_text
    return csv_text


@app.post("/similar-channels/stream")
async def similar_channels_stream(payload: SimilarChannelRequest):
    """
    流式返回相似频道搜索结果，使用 Server-Sent Events (SSE) 推送进度更新。
    """
    async def generate():
        try:
            result_data = None
            error_occurred = False
            error_message = None
            progress_queue = asyncio.Queue()
            
            def progress_callback(progress: float, message: str):
                # 将进度更新放入队列
                # 由于 get_similar_channels_by_url 是 async 函数，在同一个事件循环中运行
                # asyncio.Queue.put_nowait 是线程安全的
                try:
                    progress_queue.put_nowait((progress, message))
                except Exception:
                    pass  # 如果队列已满或其他错误，忽略
            
            # 在后台任务中执行搜索
            async def run_search():
                nonlocal result_data, error_occurred, error_message
                try:
                    result_data = await get_similar_channels_by_url(
                        channel_url=payload.channel_url,
                        max_results=payload.max_results,
                        min_subscribers=payload.min_subscribers,
                        max_subscribers=payload.max_subscribers,
                        preferred_language=payload.preferred_language,
                        preferred_region=payload.preferred_region,
                        min_similarity=payload.min_similarity,
                        progress_callback=progress_callback,
                        bd_mode=payload.bd_mode,
                    )
                except (ValueError, HTTPException) as e:
                    error_occurred = True
                    error_message = str(e) if isinstance(e, ValueError) else e.detail
                except Exception as e:
                    error_occurred = True
                    error_message = f"Internal error: {e}"
            
            # 启动搜索任务
            search_task = asyncio.create_task(run_search())
            
            # 持续监听进度更新，直到搜索完成（CP-y2-19：响应流式传输优化）
            last_heartbeat = asyncio.get_event_loop().time()
            heartbeat_interval = 30  # 每30秒发送一次心跳，防止连接超时
            
            while not search_task.done():
                try:
                    # 等待进度更新，设置超时避免阻塞
                    progress, message = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    progress_event = {
                        "type": "progress",
                        "progress": progress,
                        "message": message
                    }
                    yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"
                    last_heartbeat = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    # 超时，检查是否需要发送心跳
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        # 发送心跳保持连接
                        yield f": heartbeat\n\n"
                        last_heartbeat = current_time
                    await asyncio.sleep(0.05)
                    continue
            
            # 等待搜索任务完成
            await search_task
            
            if error_occurred:
                error_event = {
                    "type": "error",
                    "error": error_message
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                return
            
            # 发送最终结果
            if result_data:
                # 生成结果 ID 并缓存结果
                result_id = generate_result_id()
                store_result(result_id, result_data)
                result_data["result_id"] = result_id
                
                result_event = {
                    "type": "result",
                    "data": result_data
                }
                yield f"data: {json.dumps(result_event, ensure_ascii=False)}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            error_event = {
                "type": "error",
                "error": f"Internal error: {e}"
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/quota/usage")
async def quota_usage():
    """
    查询今天的配额使用情况
    """
    usage = get_quota_usage_today()
    return usage


@app.get("/quota/usage/by-endpoint")
async def quota_usage_by_endpoint(days: int = Query(default=1, ge=1, le=30, description="统计最近N天的数据")):
    """
    按端点统计配额使用情况
    """
    usage = get_quota_usage_by_endpoint(days=days)
    return usage


@app.get("/quota/status")
async def quota_status():
    """
    查询配额状态（包含是否耗尽、是否告警等）
    """
    status = get_quota_status()
    return status


@app.get("/quota/fallback-stats")
async def quota_fallback_stats(days: int = Query(default=1, ge=1, le=30, description="统计最近N天的数据")):
    """
    查询配额耗尽时的降级统计信息
    
    返回降级统计，包括：
    - total_fallbacks: 总降级次数
    - success_count: 成功降级次数
    - fail_count: 失败降级次数
    - by_type: 按降级类型统计（如 local_db_channel_info, skip_operation 等）
    - by_endpoint: 按API端点统计
    """
    stats = get_fallback_stats(days=days)
    return stats


@app.get("/quota/statistics")
async def quota_statistics(days: int = Query(default=7, ge=1, le=30, description="统计最近N天的数据")):
    """
    查询配额使用统计信息（CP-y3-10）
    
    返回详细的配额使用统计，包括：
    - total_used: 总使用配额
    - total_quota: 总配额（days * daily_quota）
    - average_daily_usage: 平均每日使用量
    - average_daily_rate: 平均每日使用率
    - peak_usage_day: 使用量最高的一天
    - peak_usage_rate: 最高使用率
    - by_endpoint: 按端点统计
    - by_day: 按天统计
    """
    stats = get_quota_statistics(days=days)
    return stats


@app.get("/quota/usage-rate")
async def quota_usage_rate():
    """
    查询配额使用率（CP-y3-02）
    
    返回当前配额使用率（0-100）
    """
    usage_rate = get_quota_usage_rate()
    return {"usage_rate": usage_rate}


@app.get("/cache/stats")
async def cache_stats():
    """
    查询结果缓存统计信息（CP-y2-06）
    
    返回缓存统计信息，包括：
    - size: 当前缓存大小
    - max_size: 最大缓存大小
    - hits: 缓存命中次数
    - misses: 缓存未命中次数
    - stores: 缓存存储次数
    - evictions: 缓存淘汰次数
    - hit_rate: 缓存命中率（%）
    - ttl_seconds: 缓存过期时间（秒）
    """
    stats = get_cache_stats()
    return stats


@app.get("/quota/rate-limit")
async def quota_rate_limit():
    """
    查询配额限流状态（CP-y3-05：配额限流机制）
    返回当前是否处于限流状态、延迟时间等信息
    """
    status = get_rate_limit_status()
    usage_rate = get_quota_usage_rate()
    status["current_usage_rate"] = usage_rate
    return status


@app.get("/quota/batch-stats")
async def quota_batch_stats():
    """
    查询批量请求统计信息（CP-y3-09：批量请求优化）
    返回批量请求使用情况、节省的配额和时间等信息
    """
    stats = get_batch_request_stats()
    return stats


@app.get("/quota/logs")
async def quota_logs(
    days: int = Query(default=1, ge=1, le=30, description="查询最近N天的配额日志"),
    limit: int = Query(default=200, ge=1, le=1000, description="返回的最大日志条数"),
    success: bool | None = Query(default=None, description="是否仅返回成功/失败的日志（None表示全部）"),
):
    """
    查询配额使用日志（CP-y3-13：配额使用日志）
    """
    logs = get_quota_usage_logs(days=days, limit=limit, success=success)
    return logs


@app.get("/quota/prediction")
async def quota_prediction(
    lookback_hours: int = Query(default=1, ge=1, le=24, description="用于计算使用速率的回看小时数（1-24）")
):
    """
    查询配额耗尽时间预测（CP-y3-11：配额预测）
    基于最近N小时的使用速率预测配额何时耗尽
    
    返回预测信息，包括：
    - current_usage_rate_per_hour: 当前使用速率（单位/小时）
    - remaining_quota: 剩余配额
    - predicted_exhaustion_time: 预测耗尽时间（ISO格式）
    - predicted_exhaustion_hours: 预测耗尽时间（小时数）
    - confidence: 预测置信度说明
    - alert_level: 告警级别（none/low/medium/high/critical）
    - alert_message: 告警消息
    """
    prediction = predict_quota_exhaustion(lookback_hours=lookback_hours)
    
    # 如果告警级别为critical或high，记录日志
    if prediction.get("alert_level") in ["critical", "high"]:
        logger.warning(f"配额预测告警: {prediction.get('alert_message')}")
    
    return prediction


@app.get("/quota/warnings")
async def quota_warnings(
    days: int = Query(default=7, ge=1, le=30, description="查询最近N天的告警"),
    unacknowledged_only: bool = Query(default=False, description="是否只返回未确认的告警"),
):
    """
    查询配额告警记录（CP-y3-04）
    
    返回配额告警列表，包括：
    - id: 告警ID
    - warning_type: 告警类型
    - usage_rate: 使用率
    - used_quota: 已使用配额
    - total_quota: 总配额
    - message: 告警消息
    - timestamp: 告警时间
    - acknowledged: 是否已确认
    """
    warnings = get_quota_warnings(days=days, unacknowledged_only=unacknowledged_only)
    return {"warnings": warnings, "count": len(warnings)}


@app.post("/quota/clean-legacy")
async def clean_legacy_quota_records_endpoint(
    dry_run: bool = Query(default=False, description="试运行模式，只统计不删除")
):
    """
    清理添加 use_for 字段之前的旧记录（use_for IS NULL 的记录）
    
    这些旧记录是在添加多 API Key 支持之前创建的，会影响默认用途的配额统计。
    清理后，默认用途（use_for IS NULL）的统计将只包含真正使用默认 API Key 的记录。
    
    Args:
        dry_run: 如果为 True，只统计要删除的记录数，不实际删除
    
    Returns:
        包含清理统计信息的字典
    """
    result = clean_legacy_quota_records(dry_run=dry_run)
    return result


@app.post("/quota/warnings/{warning_id}/acknowledge")
async def acknowledge_warning(warning_id: int):
    """
    确认配额告警（CP-y3-04）
    
    将指定的告警标记为已处理
    """
    success = acknowledge_quota_warning(warning_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"告警 ID {warning_id} 不存在")
    return {"success": True, "message": f"告警 {warning_id} 已确认"}


