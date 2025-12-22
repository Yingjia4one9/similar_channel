"""
指标统计模块
收集和统计系统指标，包括性能监控、业务指标等
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger()


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """性能指标"""
    operation: str
    duration: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器（单例模式）"""
    
    _instance: Optional['MetricsCollector'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._lock = Lock()
        self._performance_metrics: List[PerformanceMetric] = []
        self._business_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._max_metrics_history = 10000  # 最多保留10000条性能指标
        self._max_points_per_metric = 1000  # 每个业务指标最多保留1000个数据点
        self._initialized = True
    
    def record_performance(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录性能指标
        
        Args:
            operation: 操作名称（如 "search_channels", "get_channel_info"）
            duration: 操作耗时（秒）
            success: 是否成功
            metadata: 额外的元数据
        """
        with self._lock:
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                timestamp=datetime.now(),
                success=success,
                metadata=metadata or {},
            )
            self._performance_metrics.append(metric)
            
            # 限制历史记录数量
            if len(self._performance_metrics) > self._max_metrics_history:
                self._performance_metrics = self._performance_metrics[-self._max_metrics_history:]
    
    def record_business_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        记录业务指标
        
        Args:
            metric_name: 指标名称（如 "channels_found", "similarity_score"）
            value: 指标值
            tags: 标签（用于分类，如 {"source": "local_db"}）
        """
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {},
            )
            self._business_metrics[metric_name].append(point)
            
            # 限制每个指标的数据点数量
            if len(self._business_metrics[metric_name]) > self._max_points_per_metric:
                self._business_metrics[metric_name] = self._business_metrics[metric_name][-self._max_points_per_metric:]
    
    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """
        增加计数器
        
        Args:
            counter_name: 计数器名称（如 "api_calls", "cache_hits"）
            value: 增加值（默认为1）
        """
        with self._lock:
            self._counters[counter_name] += value
    
    def get_performance_stats(
        self,
        operation: Optional[str] = None,
        hours: int = 24,
        success_only: bool = False,
    ) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Args:
            operation: 操作名称（如果为None，则统计所有操作）
            hours: 统计最近N小时的数据
            success_only: 是否只统计成功的操作
            
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = [
                m for m in self._performance_metrics
                if m.timestamp >= cutoff_time
                and (operation is None or m.operation == operation)
                and (not success_only or m.success)
            ]
            
            if not metrics:
                return {
                    "count": 0,
                    "avg_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                    "p50_duration": 0.0,
                    "p95_duration": 0.0,
                    "p99_duration": 0.0,
                    "success_rate": 0.0,
                }
            
            durations = [m.duration for m in metrics]
            durations.sort()
            
            success_count = sum(1 for m in metrics if m.success)
            
            def percentile(data: List[float], p: float) -> float:
                """计算百分位数"""
                if not data:
                    return 0.0
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return data[f] + c * (data[f + 1] - data[f])
                return data[f]
            
            return {
                "count": len(metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0.0,
                "min_duration": min(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0,
                "p50_duration": percentile(durations, 0.50),
                "p95_duration": percentile(durations, 0.95),
                "p99_duration": percentile(durations, 0.99),
                "success_rate": success_count / len(metrics) if metrics else 0.0,
            }
    
    def get_business_metric_stats(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        获取业务指标统计信息
        
        Args:
            metric_name: 指标名称
            hours: 统计最近N小时的数据
            
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            points = [
                p for p in self._business_metrics.get(metric_name, [])
                if p.timestamp >= cutoff_time
            ]
            
            if not points:
                return {
                    "count": 0,
                    "avg_value": 0.0,
                    "min_value": 0.0,
                    "max_value": 0.0,
                }
            
            values = [p.value for p in points]
            
            return {
                "count": len(points),
                "avg_value": sum(values) / len(values) if values else 0.0,
                "min_value": min(values) if values else 0.0,
                "max_value": max(values) if values else 0.0,
            }
    
    def get_counter_value(self, counter_name: str) -> int:
        """
        获取计数器当前值
        
        Args:
            counter_name: 计数器名称
            
        Returns:
            计数器值
        """
        with self._lock:
            return self._counters.get(counter_name, 0)
    
    def reset_counter(self, counter_name: str) -> None:
        """
        重置计数器
        
        Args:
            counter_name: 计数器名称
        """
        with self._lock:
            self._counters[counter_name] = 0
    
    def get_all_counters(self) -> Dict[str, int]:
        """
        获取所有计数器的值
        
        Returns:
            计数器字典
        """
        with self._lock:
            return dict(self._counters)
    
    def clear_old_metrics(self, days: int = 7) -> None:
        """
        清理旧的指标数据
        
        Args:
            days: 保留最近N天的数据
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # 清理性能指标
            self._performance_metrics = [
                m for m in self._performance_metrics
                if m.timestamp >= cutoff_time
            ]
            
            # 清理业务指标
            for metric_name in list(self._business_metrics.keys()):
                self._business_metrics[metric_name] = [
                    p for p in self._business_metrics[metric_name]
                    if p.timestamp >= cutoff_time
                ]
                
                # 如果指标为空，删除它
                if not self._business_metrics[metric_name]:
                    del self._business_metrics[metric_name]


# 全局指标收集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例（单例）"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# 便捷函数
def record_performance(
    operation: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """记录性能指标（便捷函数）"""
    get_metrics_collector().record_performance(operation, duration, success, metadata)


def record_business_metric(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """记录业务指标（便捷函数）"""
    get_metrics_collector().record_business_metric(metric_name, value, tags)


def increment_counter(counter_name: str, value: int = 1) -> None:
    """增加计数器（便捷函数）"""
    get_metrics_collector().increment_counter(counter_name, value)


def performance_timer(operation: str):
    """
    性能计时装饰器
    
    Usage:
        @performance_timer("get_channel_info")
        def get_channel_info(channel_id):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                record_performance(operation, duration, success)
        return wrapper
    return decorator

