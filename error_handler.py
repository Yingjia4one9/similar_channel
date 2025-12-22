"""
统一错误处理模块（CP-y6-10：错误处理统一）
提供统一的错误处理函数和异常类
"""
from typing import Any, Optional
from logger import get_logger

logger = get_logger()


def handle_exception(
    exception: Exception,
    context: str = "",
    log_level: str = "error",
    reraise: bool = True,
    custom_message: Optional[str] = None
) -> None:
    """
    统一处理异常（CP-y6-10：错误处理统一）
    
    Args:
        exception: 要处理的异常
        context: 异常发生的上下文描述
        log_level: 日志级别（"error", "warning", "info", "debug"）
        reraise: 是否重新抛出异常
        custom_message: 自定义错误消息（如果为None，则使用异常消息）
    
    Raises:
        原异常（如果reraise=True）
    """
    error_msg = custom_message or str(exception)
    log_msg = f"{context}: {error_msg}" if context else error_msg
    
    # 根据日志级别记录
    if log_level == "error":
        logger.error(log_msg, exc_info=True)
    elif log_level == "warning":
        logger.warning(log_msg, exc_info=True)
    elif log_level == "info":
        logger.info(log_msg)
    elif log_level == "debug":
        logger.debug(log_msg)
    else:
        logger.error(log_msg, exc_info=True)
    
    # 如果需要，重新抛出异常
    if reraise:
        raise


def safe_execute(
    func,
    *args,
    default_return: Any = None,
    context: str = "",
    log_level: str = "warning",
    **kwargs
) -> Any:
    """
    安全执行函数，捕获所有异常并返回默认值（CP-y6-10：错误处理统一）
    
    Args:
        func: 要执行的函数
        *args: 函数的位置参数
        default_return: 发生异常时返回的默认值
        context: 执行上下文描述
        log_level: 日志级别
        **kwargs: 函数的关键字参数
    
    Returns:
        函数执行结果，或发生异常时返回default_return
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_exception(
            e,
            context=context or f"执行函数 {func.__name__}",
            log_level=log_level,
            reraise=False
        )
        return default_return

