"""
嵌入向量计算模块
处理文本向量化、标签推理等功能
优化版：支持标签互斥组、分层阈值、预计算归一化缓存

异步优化：CPU 密集型任务（模型编码）使用线程池执行，避免阻塞事件循环
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from infrastructure.config import Config


# 全局模型实例（延迟加载）
_embed_model: SentenceTransformer | None = None

# 标签向量缓存
_topic_embeds: np.ndarray | None = None
_audience_embeds: np.ndarray | None = None

# 预计算的归一化向量缓存（优化计算性能）
_topic_embeds_norm: np.ndarray | None = None
_audience_embeds_norm: np.ndarray | None = None

# 标签索引映射（用于快速查找）
_topic_label_to_idx: Dict[str, int] | None = None
_audience_label_to_idx: Dict[str, int] | None = None

# ==================== 线程池管理器（用于 CPU 密集型任务）====================
# 全局线程池，用于执行模型编码等 CPU 密集型任务
_executor: ThreadPoolExecutor | None = None

# 向量编码并发控制信号量（限制同时进行向量编码的线程数，避免内存不足）
# 默认最多 2 个线程同时进行向量编码，减少内存压力
# 使用 threading.Semaphore 以便跨线程和事件循环共享
_encoding_semaphore: threading.Semaphore | None = None
_encoding_semaphore_lock = threading.Lock()


def _get_encoding_semaphore() -> threading.Semaphore:
    """
    获取向量编码信号量（单例模式）。
    用于限制同时进行向量编码的线程数，避免内存不足。
    使用 threading.Semaphore 以便跨线程和事件循环共享。
    
    Returns:
        threading.Semaphore 实例
    """
    global _encoding_semaphore
    if _encoding_semaphore is None:
        with _encoding_semaphore_lock:
            if _encoding_semaphore is None:
                # 从配置获取最大并发数，默认 2（减少内存压力）
                from infrastructure.config import Config
                max_concurrent = Config.get_config_value(
                    "embedding.max_concurrent_encoding",
                    2,  # 默认最多 2 个线程同时编码，避免内存不足
                    "YT_EMBEDDING_MAX_CONCURRENT"
                )
                # 使用 threading.Semaphore 以便跨线程共享
                _encoding_semaphore = threading.Semaphore(max_concurrent)
    return _encoding_semaphore


def _get_executor() -> ThreadPoolExecutor:
    """
    获取全局线程池实例（单例模式）。
    线程池大小根据配置和 CPU 核心数自动调整。
    
    Returns:
        ThreadPoolExecutor 实例
    """
    global _executor
    if _executor is None:
        # 从配置获取线程数，默认使用 CPU 核心数
        from infrastructure.config import Config
        max_workers = Config.get_config_value(
            "embedding.thread_pool_size",
            Config.CONCURRENT_PROCESSING.get("search_workers", 4),
            "YT_EMBEDDING_THREAD_POOL_SIZE"
        )
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="embedding")
    return _executor


def _sync_encode(model: SentenceTransformer, texts: List[str], **kwargs) -> np.ndarray:
    """
    同步编码函数，在线程池中执行。
    这是 model.encode 的包装函数，用于在线程池中运行。
    
    Args:
        model: SentenceTransformer 模型实例
        texts: 要编码的文本列表
        **kwargs: 传递给 model.encode 的其他参数
    
    Returns:
        编码后的向量数组
    """
    return model.encode(texts, convert_to_numpy=True, **kwargs)


async def encode_async(texts: List[str], **kwargs) -> np.ndarray:
    """
    异步编码函数，在线程池中执行模型编码，避免阻塞事件循环。
    使用信号量限制并发数，避免内存不足。
    
    Args:
        texts: 要编码的文本列表
        **kwargs: 传递给 model.encode 的其他参数（如 batch_size, show_progress_bar）
    
    Returns:
        编码后的向量数组
    
    Example:
        vecs = await encode_async(["text1", "text2"], batch_size=32)
    """
    if not texts:
        return np.array([])
    
    # 获取信号量，限制同时进行编码的线程数
    semaphore = _get_encoding_semaphore()
    
    # 使用 threading.Semaphore，需要手动获取和释放
    semaphore.acquire()
    try:
        # #region agent log
        try:
            import json
            import time
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"core/embedding.py:encode_async:ACQUIRE","message":"获取编码信号量","data":{"texts_count":len(texts),"semaphore_value":semaphore._value},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        
        model = get_embed_model()
        loop = asyncio.get_running_loop()
        executor = _get_executor()
        
        # 使用 functools.partial 来传递 kwargs，因为 run_in_executor 不接受关键字参数
        from functools import partial
        sync_encode_with_kwargs = partial(_sync_encode, model, texts, **kwargs)
        
        try:
            # 在线程池中执行编码任务
            result = await loop.run_in_executor(executor, sync_encode_with_kwargs)
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"core/embedding.py:encode_async:SUCCESS","message":"编码成功","data":{"texts_count":len(texts),"result_shape":list(result.shape) if hasattr(result,'shape') else None},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            return result
        except OSError as e:
            # #region agent log
            try:
                with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"core/embedding.py:encode_async:OS_ERROR","message":"编码时内存错误","data":{"error":str(e),"error_type":type(e).__name__,"error_code":getattr(e,'winerror',None)},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            # 内存不足错误，重新抛出（信号量会在 finally 中释放）
            raise
    finally:
        # 确保释放信号量
        semaphore.release()
        # #region agent log
        try:
            with open(r'c:\Users\A\Desktop\yt-similar-backend\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"core/embedding.py:encode_async:RELEASE","message":"释放编码信号量","data":{"texts_count":len(texts)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion


def get_embed_model() -> SentenceTransformer:
    """
    获取嵌入模型实例（单例模式，优化版）。
    Sentence Transformers 5.x 支持更好的设备管理和性能优化。
    
    Returns:
        SentenceTransformer 模型实例
    """
    global _embed_model
    if _embed_model is None:
        # 多语言句向量模型：兼顾效果与资源占用
        # Sentence Transformers 5.x 支持更明确的设备设置和批处理优化
        _embed_model = SentenceTransformer(
            Config.EMBED_MODEL_NAME,
            device='cpu',  # 明确指定设备，如果未来有 GPU 可以改为 'cuda'
        )
        # 5.x 版本支持模型预热，提高首次使用速度
        try:
            _embed_model.encode(['warmup'], convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            pass  # 如果预热失败，不影响使用
    return _embed_model


def ensure_label_embeddings(model: SentenceTransformer) -> None:
    """
    确保为 Topics / Audience 标签计算好向量，只计算一次复用。
    优化版：预计算归一化向量和索引映射。
    
    Args:
        model: 嵌入模型实例
    """
    global _topic_embeds, _audience_embeds
    global _topic_embeds_norm, _audience_embeds_norm
    global _topic_label_to_idx, _audience_label_to_idx
    
    # 使用配置的批量大小
    batch_size = Config.get_config_value("embedding.batch_size", Config.EMBEDDING_BATCH_SIZE)
    
    if _topic_embeds is None:
        _topic_embeds = model.encode(
            Config.TOPIC_LABELS,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        # 预计算归一化向量（避免每次推理时重复计算）
        topic_norms = np.linalg.norm(_topic_embeds, axis=1, keepdims=True)
        _topic_embeds_norm = _topic_embeds / (topic_norms + 1e-8)
        # 构建标签索引映射
        _topic_label_to_idx = {label: idx for idx, label in enumerate(Config.TOPIC_LABELS)}
        
    if _audience_embeds is None:
        _audience_embeds = model.encode(
            Config.AUDIENCE_LABELS,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        # 预计算归一化向量
        audience_norms = np.linalg.norm(_audience_embeds, axis=1, keepdims=True)
        _audience_embeds_norm = _audience_embeds / (audience_norms + 1e-8)
        # 构建标签索引映射
        _audience_label_to_idx = {label: idx for idx, label in enumerate(Config.AUDIENCE_LABELS)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度（优化版，使用 NumPy 2.x 特性）。
    
    Args:
        a: 第一个向量
        b: 第二个向量
    
    Returns:
        余弦相似度值（0-1）
    """
    # 展平向量（NumPy 2.x 优化的 flatten 操作）
    a = a.flatten() if a.ndim > 1 else a
    b = b.flatten() if b.ndim > 1 else b
    
    # 使用 NumPy 2.x 优化的点积和范数计算
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    
    if norm_product == 0:
        return 0.0
    
    return float(dot_product / norm_product)


def _apply_mutual_exclusion(
    labels_with_scores: List[Tuple[str, float]], 
    exclusion_groups: List[List[str]]
) -> List[str]:
    """
    应用标签互斥组规则：同组标签只保留相似度最高的一个。
    
    Args:
        labels_with_scores: (标签, 相似度分数) 的列表，已按分数降序排列
        exclusion_groups: 互斥组列表
    
    Returns:
        去重后的标签列表
    """
    if not exclusion_groups:
        return [label for label, _ in labels_with_scores]
    
    # 构建标签到互斥组的映射
    label_to_group: Dict[str, int] = {}
    for group_idx, group in enumerate(exclusion_groups):
        for label in group:
            label_to_group[label] = group_idx
    
    # 已选中的互斥组
    selected_groups: Set[int] = set()
    result: List[str] = []
    
    for label, score in labels_with_scores:
        group_idx = label_to_group.get(label)
        
        if group_idx is not None:
            # 标签属于某个互斥组
            if group_idx not in selected_groups:
                # 该组还没有选中标签，选择这个（分数最高的）
                result.append(label)
                selected_groups.add(group_idx)
            # 否则跳过（该组已有更高分数的标签）
        else:
            # 标签不属于任何互斥组，直接加入
            result.append(label)
    
    return result


def _get_threshold_for_label(label: str, core_labels: List[str], 
                             core_threshold: float, extended_threshold: float) -> float:
    """
    根据标签是否为核心标签返回对应的阈值。
    
    Args:
        label: 标签名
        core_labels: 核心标签列表
        core_threshold: 核心标签阈值
        extended_threshold: 扩展标签阈值
    
    Returns:
        对应的阈值
    """
    return core_threshold if label in core_labels else extended_threshold


def infer_topics_and_audience(text_vec: np.ndarray) -> Union[Dict[str, List[str]], List[Dict[str, List[str]]]]:
    """
    基于频道整体文本向量，为其打上 Topics / Audience 标签。
    优化版：支持分层阈值、标签互斥组、预计算归一化缓存。
    
    Args:
        text_vec: 频道文本的向量表示（可以是单个向量或向量数组）
    
    Returns:
        包含 "topics" 和 "audience" 两个键的字典（单个）或字典列表（批量）
    """
    # 如果输入是批量向量，使用批量处理
    if text_vec.ndim > 1 and text_vec.shape[0] > 1:
        return _infer_topics_and_audience_batch(text_vec)
    
    # 单个向量处理（保持向后兼容）
    if text_vec.ndim > 1:
        text_vec = text_vec[0]

    topics: List[str] = []
    audiences: List[str] = []
    
    # 从配置读取参数
    max_topics = Config.TAG_INFERENCE.get("max_topics", 10)
    max_audience = Config.TAG_INFERENCE.get("max_audience", 8)
    core_threshold = Config.TAG_INFERENCE.get("core_threshold", 0.35)
    extended_threshold = Config.TAG_INFERENCE.get("extended_threshold", 0.25)
    enable_mutual_exclusion = Config.TAG_INFERENCE.get("enable_mutual_exclusion", True)
    
    # 归一化输入向量
    text_vec_norm = text_vec / (np.linalg.norm(text_vec) + 1e-8)

    # 处理 Topics 标签
    if _topic_embeds_norm is not None:
        # 使用预计算的归一化向量计算相似度
        sims = _topic_embeds_norm @ text_vec_norm
        
        # 收集所有超过阈值的标签及其分数
        topic_candidates: List[Tuple[str, float]] = []
        for idx in range(len(Config.TOPIC_LABELS)):
            label = Config.TOPIC_LABELS[idx]
            score = float(sims[idx])
            # 使用分层阈值
            threshold = _get_threshold_for_label(
                label, Config.CORE_TOPIC_LABELS, core_threshold, extended_threshold
            )
            if score >= threshold:
                topic_candidates.append((label, score))
        
        # 按分数降序排列
        topic_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 应用互斥组规则
        if enable_mutual_exclusion:
            topics = _apply_mutual_exclusion(
                topic_candidates[:max_topics * 2],  # 取更多候选以便互斥后仍有足够数量
                Config.TOPIC_MUTUAL_EXCLUSION_GROUPS
            )[:max_topics]
        else:
            topics = [label for label, _ in topic_candidates[:max_topics]]

    # 处理 Audience 标签
    if _audience_embeds_norm is not None:
        # 使用预计算的归一化向量计算相似度
        sims = _audience_embeds_norm @ text_vec_norm
        
        # 收集所有超过阈值的标签及其分数
        audience_candidates: List[Tuple[str, float]] = []
        for idx in range(len(Config.AUDIENCE_LABELS)):
            label = Config.AUDIENCE_LABELS[idx]
            score = float(sims[idx])
            # 使用分层阈值
            threshold = _get_threshold_for_label(
                label, Config.CORE_AUDIENCE_LABELS, core_threshold, extended_threshold
            )
            if score >= threshold:
                audience_candidates.append((label, score))
        
        # 按分数降序排列
        audience_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 应用互斥组规则
        if enable_mutual_exclusion:
            audiences = _apply_mutual_exclusion(
                audience_candidates[:max_audience * 2],
                Config.AUDIENCE_MUTUAL_EXCLUSION_GROUPS
            )[:max_audience]
        else:
            audiences = [label for label, _ in audience_candidates[:max_audience]]

    return {"topics": topics, "audience": audiences}


def _infer_topics_and_audience_batch(text_vecs: np.ndarray) -> List[Dict[str, List[str]]]:
    """
    批量处理多个向量的标签推理（内部函数，优化版）。
    支持分层阈值、标签互斥组、预计算归一化缓存。
    
    Args:
        text_vecs: 多个频道文本向量的数组，形状为 (n, embedding_dim)
    
    Returns:
        字典列表，每个字典包含 "topics" 和 "audience" 两个键
    """
    results: List[Dict[str, List[str]]] = []
    
    # 从配置读取参数
    max_topics = Config.TAG_INFERENCE.get("max_topics", 10)
    max_audience = Config.TAG_INFERENCE.get("max_audience", 8)
    core_threshold = Config.TAG_INFERENCE.get("core_threshold", 0.35)
    extended_threshold = Config.TAG_INFERENCE.get("extended_threshold", 0.25)
    enable_mutual_exclusion = Config.TAG_INFERENCE.get("enable_mutual_exclusion", True)
    
    # 批量归一化所有向量（一次性处理，提高效率）
    text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
    text_vecs_norm = text_vecs / (text_norms + 1e-8)
    
    # 批量计算相似度（使用预计算的归一化向量）
    topic_sims = None
    audience_sims = None
    
    if _topic_embeds_norm is not None:
        # (n, embedding_dim) @ (n_topics, embedding_dim).T -> (n, n_topics)
        topic_sims = text_vecs_norm @ _topic_embeds_norm.T
    
    if _audience_embeds_norm is not None:
        # (n, embedding_dim) @ (n_audience, embedding_dim).T -> (n, n_audience)
        audience_sims = text_vecs_norm @ _audience_embeds_norm.T
    
    # 为每个向量生成标签
    for i in range(text_vecs.shape[0]):
        topics: List[str] = []
        audiences: List[str] = []
        
        # 处理 Topics
        if topic_sims is not None:
            sims = topic_sims[i]
            
            # 收集所有超过阈值的标签及其分数
            topic_candidates: List[Tuple[str, float]] = []
            for idx in range(len(Config.TOPIC_LABELS)):
                label = Config.TOPIC_LABELS[idx]
                score = float(sims[idx])
                threshold = _get_threshold_for_label(
                    label, Config.CORE_TOPIC_LABELS, core_threshold, extended_threshold
                )
                if score >= threshold:
                    topic_candidates.append((label, score))
            
            # 按分数降序排列
            topic_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 应用互斥组规则
            if enable_mutual_exclusion:
                topics = _apply_mutual_exclusion(
                    topic_candidates[:max_topics * 2],
                    Config.TOPIC_MUTUAL_EXCLUSION_GROUPS
                )[:max_topics]
            else:
                topics = [label for label, _ in topic_candidates[:max_topics]]
        
        # 处理 Audience
        if audience_sims is not None:
            sims = audience_sims[i]
            
            # 收集所有超过阈值的标签及其分数
            audience_candidates: List[Tuple[str, float]] = []
            for idx in range(len(Config.AUDIENCE_LABELS)):
                label = Config.AUDIENCE_LABELS[idx]
                score = float(sims[idx])
                threshold = _get_threshold_for_label(
                    label, Config.CORE_AUDIENCE_LABELS, core_threshold, extended_threshold
                )
                if score >= threshold:
                    audience_candidates.append((label, score))
            
            # 按分数降序排列
            audience_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 应用互斥组规则
            if enable_mutual_exclusion:
                audiences = _apply_mutual_exclusion(
                    audience_candidates[:max_audience * 2],
                    Config.AUDIENCE_MUTUAL_EXCLUSION_GROUPS
                )[:max_audience]
            else:
                audiences = [label for label, _ in audience_candidates[:max_audience]]
        
        results.append({"topics": topics, "audience": audiences})
    
    return results


def infer_topics_and_audience_with_scores(text_vec: np.ndarray) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """
    基于频道整体文本向量，为其打上带置信度分数的标签。
    用于需要更精细控制的场景（如相似度计算时加权）。
    
    Args:
        text_vec: 频道文本的向量表示
    
    Returns:
        包含 "topics" 和 "audience" 两个键的字典，
        每个值是 [{"label": str, "confidence": float}, ...] 列表
    """
    if text_vec.ndim > 1:
        text_vec = text_vec[0]
    
    topics_with_scores: List[Dict[str, Union[str, float]]] = []
    audiences_with_scores: List[Dict[str, Union[str, float]]] = []
    
    # 从配置读取参数
    max_topics = Config.TAG_INFERENCE.get("max_topics", 10)
    max_audience = Config.TAG_INFERENCE.get("max_audience", 8)
    core_threshold = Config.TAG_INFERENCE.get("core_threshold", 0.35)
    extended_threshold = Config.TAG_INFERENCE.get("extended_threshold", 0.25)
    enable_mutual_exclusion = Config.TAG_INFERENCE.get("enable_mutual_exclusion", True)
    
    # 归一化输入向量
    text_vec_norm = text_vec / (np.linalg.norm(text_vec) + 1e-8)

    # 处理 Topics 标签
    if _topic_embeds_norm is not None:
        sims = _topic_embeds_norm @ text_vec_norm
        
        topic_candidates: List[Tuple[str, float]] = []
        for idx in range(len(Config.TOPIC_LABELS)):
            label = Config.TOPIC_LABELS[idx]
            score = float(sims[idx])
            threshold = _get_threshold_for_label(
                label, Config.CORE_TOPIC_LABELS, core_threshold, extended_threshold
            )
            if score >= threshold:
                topic_candidates.append((label, score))
        
        topic_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if enable_mutual_exclusion:
            filtered_labels = _apply_mutual_exclusion(
                topic_candidates[:max_topics * 2],
                Config.TOPIC_MUTUAL_EXCLUSION_GROUPS
            )[:max_topics]
            # 保留分数
            label_scores = {label: score for label, score in topic_candidates}
            topics_with_scores = [
                {"label": label, "confidence": round(label_scores[label], 3)}
                for label in filtered_labels
            ]
        else:
            topics_with_scores = [
                {"label": label, "confidence": round(score, 3)}
                for label, score in topic_candidates[:max_topics]
            ]

    # 处理 Audience 标签
    if _audience_embeds_norm is not None:
        sims = _audience_embeds_norm @ text_vec_norm
        
        audience_candidates: List[Tuple[str, float]] = []
        for idx in range(len(Config.AUDIENCE_LABELS)):
            label = Config.AUDIENCE_LABELS[idx]
            score = float(sims[idx])
            threshold = _get_threshold_for_label(
                label, Config.CORE_AUDIENCE_LABELS, core_threshold, extended_threshold
            )
            if score >= threshold:
                audience_candidates.append((label, score))
        
        audience_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if enable_mutual_exclusion:
            filtered_labels = _apply_mutual_exclusion(
                audience_candidates[:max_audience * 2],
                Config.AUDIENCE_MUTUAL_EXCLUSION_GROUPS
            )[:max_audience]
            label_scores = {label: score for label, score in audience_candidates}
            audiences_with_scores = [
                {"label": label, "confidence": round(label_scores[label], 3)}
                for label in filtered_labels
            ]
        else:
            audiences_with_scores = [
                {"label": label, "confidence": round(score, 3)}
                for label, score in audience_candidates[:max_audience]
            ]

    return {"topics": topics_with_scores, "audience": audiences_with_scores}
