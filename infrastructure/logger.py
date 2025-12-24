"""
日志管理模块
统一管理应用的日志记录
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "yt_similar",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    设置并返回配置好的 logger 实例。
    
    Args:
        name: logger 名称
        level: 日志级别
        format_string: 自定义格式字符串
    
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 创建控制台 handler（强制刷新输出，避免缓冲导致看不到实时日志）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # 设置流为无缓冲模式，确保实时输出
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except: pass
    
    # 设置格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# 全局 logger 实例
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    获取全局 logger 实例（单例模式）。
    
    Returns:
        logger 实例
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger

