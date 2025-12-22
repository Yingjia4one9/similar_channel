"""
嵌入向量计算模块
处理文本向量化、标签推理等功能
"""
from typing import Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config


# 全局模型实例（延迟加载）
_embed_model: SentenceTransformer | None = None

# 标签向量缓存
_topic_embeds: np.ndarray | None = None
_audience_embeds: np.ndarray | None = None


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
    Sentence Transformers 5.x 支持更好的批处理配置。
    
    Args:
        model: 嵌入模型实例
    """
    global _topic_embeds, _audience_embeds
    # 使用配置的批量大小
    batch_size = Config.get_config_value("embedding.batch_size", Config.EMBEDDING_BATCH_SIZE)
    
    if _topic_embeds is None:
        _topic_embeds = model.encode(
            Config.TOPIC_LABELS,
            convert_to_numpy=True,
            batch_size=batch_size,  # 使用配置的批量大小
            show_progress_bar=False,
        )
    if _audience_embeds is None:
        _audience_embeds = model.encode(
            Config.AUDIENCE_LABELS,
            convert_to_numpy=True,
            batch_size=batch_size,  # 使用配置的批量大小
            show_progress_bar=False,
        )


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


def infer_topics_and_audience(text_vec: np.ndarray) -> Union[Dict[str, List[str]], List[Dict[str, List[str]]]]:
    """
    基于频道整体文本向量，为其打上 Topics / Audience 标签。
    简单做法：把频道向量与预设标签向量做相似度，取相似度最高的若干个。
    
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

    if _topic_embeds is not None:
        sims = _topic_embeds @ text_vec / (
            np.linalg.norm(_topic_embeds, axis=1) * (np.linalg.norm(text_vec) + 1e-8)
        )
        # 动态阈值：以最高相似度为基准，保留若干个"足够接近"的主题，避免只贴 1~2 个标签。
        if sims.size > 0:
            max_sim = float(np.max(sims))
        else:
            max_sim = 0.0
        # 至少要达到基础阈值，同时不低于最高相似度的比例
        topic_threshold = max(Config.TAG_THRESHOLD_BASE, max_sim * Config.TAG_THRESHOLD_RATIO)
        top_idx = np.argsort(-sims)[:5]
        for idx in top_idx:
            # 边界检查：确保索引在有效范围内
            if 0 <= idx < len(Config.TOPIC_LABELS) and sims[idx] >= topic_threshold:
                topics.append(Config.TOPIC_LABELS[idx])

    if _audience_embeds is not None:
        sims = _audience_embeds @ text_vec / (
            np.linalg.norm(_audience_embeds, axis=1) * (np.linalg.norm(text_vec) + 1e-8)
        )
        if sims.size > 0:
            max_sim = float(np.max(sims))
        else:
            max_sim = 0.0
        # 受众画像同样采用动态阈值，多贴一些"相近人群"，便于后面做重合度匹配。
        audience_threshold = max(Config.TAG_THRESHOLD_BASE, max_sim * Config.TAG_THRESHOLD_RATIO)
        top_idx = np.argsort(-sims)[:6]
        for idx in top_idx:
            # 边界检查：确保索引在有效范围内
            if 0 <= idx < len(Config.AUDIENCE_LABELS) and sims[idx] >= audience_threshold:
                audiences.append(Config.AUDIENCE_LABELS[idx])

    return {"topics": topics, "audience": audiences}


def _infer_topics_and_audience_batch(text_vecs: np.ndarray) -> List[Dict[str, List[str]]]:
    """
    批量处理多个向量的标签推理（内部函数，优化版）。
    使用 NumPy 2.x 的向量化操作提高性能。
    
    Args:
        text_vecs: 多个频道文本向量的数组，形状为 (n, embedding_dim)
    
    Returns:
        字典列表，每个字典包含 "topics" 和 "audience" 两个键
    """
    results: List[Dict[str, List[str]]] = []
    
    # 批量归一化所有向量（一次性处理，提高效率）
    text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
    text_vecs_norm = text_vecs / (text_norms + 1e-8)
    
    # 批量计算相似度（归一化后的点积 = 余弦相似度）
    if _topic_embeds is not None:
        # 归一化标签向量
        topic_norms = np.linalg.norm(_topic_embeds, axis=1, keepdims=True)
        topic_embeds_norm = _topic_embeds / (topic_norms + 1e-8)
        # (n, embedding_dim) @ (n_topics, embedding_dim).T -> (n, n_topics)
        topic_sims = text_vecs_norm @ topic_embeds_norm.T
    else:
        topic_sims = None
    
    if _audience_embeds is not None:
        # 归一化标签向量
        audience_norms = np.linalg.norm(_audience_embeds, axis=1, keepdims=True)
        audience_embeds_norm = _audience_embeds / (audience_norms + 1e-8)
        # (n, embedding_dim) @ (n_audience, embedding_dim).T -> (n, n_audience)
        audience_sims = text_vecs_norm @ audience_embeds_norm.T
    else:
        audience_sims = None
    
    # 为每个向量生成标签
    for i in range(text_vecs.shape[0]):
        topics: List[str] = []
        audiences: List[str] = []
        
        if topic_sims is not None:
            sims = topic_sims[i]
            if sims.size > 0:
                max_sim = float(np.max(sims))
                topic_threshold = max(Config.TAG_THRESHOLD_BASE, max_sim * Config.TAG_THRESHOLD_RATIO)
                top_idx = np.argsort(-sims)[:5]
                for idx in top_idx:
                    # 边界检查：确保索引在有效范围内
                    if 0 <= idx < len(Config.TOPIC_LABELS) and sims[idx] >= topic_threshold:
                        topics.append(Config.TOPIC_LABELS[idx])
        
        if audience_sims is not None:
            sims = audience_sims[i]
            if sims.size > 0:
                max_sim = float(np.max(sims))
                audience_threshold = max(Config.TAG_THRESHOLD_BASE, max_sim * Config.TAG_THRESHOLD_RATIO)
                top_idx = np.argsort(-sims)[:6]
                for idx in top_idx:
                    # 边界检查：确保索引在有效范围内
                    if 0 <= idx < len(Config.AUDIENCE_LABELS) and sims[idx] >= audience_threshold:
                        audiences.append(Config.AUDIENCE_LABELS[idx])
        
        results.append({"topics": topics, "audience": audiences})
    
    return results

