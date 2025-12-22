"""
XSS防护模块
提供XSS检测和防护功能
"""
import re
from typing import Any, Dict, List

from logger import get_logger

logger = get_logger()

# XSS攻击模式（常见payload）
XSS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # <script>标签
    r'javascript:',  # javascript:协议
    r'on\w+\s*=',  # 事件处理器（onclick, onerror等）
    r'<iframe[^>]*>',  # iframe标签
    r'<img[^>]*onerror',  # img onerror
    r'<svg[^>]*onload',  # svg onload
    r'<body[^>]*onload',  # body onload
    r'eval\s*\(',  # eval函数
    r'expression\s*\(',  # CSS expression
    r'vbscript:',  # vbscript协议
    r'data:text/html',  # data URI with HTML
]


def detect_xss_attempt(text: str) -> bool:
    """
    检测文本中是否包含XSS攻击尝试
    
    Args:
        text: 要检测的文本
    
    Returns:
        如果检测到XSS尝试则返回True
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    for pattern in XSS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return True
    
    return False


def sanitize_user_input(text: str, log_attempts: bool = True) -> str:
    """
    清理用户输入，移除潜在的XSS代码
    
    Args:
        text: 用户输入文本
        log_attempts: 是否记录XSS尝试
    
    Returns:
        清理后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 检测XSS尝试
    if detect_xss_attempt(text):
        if log_attempts:
            # 脱敏处理，只记录前50个字符
            safe_text = text[:50] + "..." if len(text) > 50 else text
            logger.warning(f"检测到潜在的XSS攻击尝试（已清理）: {safe_text}")
        
        # 移除危险字符和标签
        # 移除<script>标签
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # 移除事件处理器
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        # 移除javascript:协议
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        # 移除vbscript:协议
        text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
        # 移除eval调用
        text = re.sub(r'eval\s*\(', '', text, flags=re.IGNORECASE)
    
    return text


def validate_and_sanitize_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证和清理请求payload，检测XSS尝试
    
    Args:
        payload: 请求payload字典
    
    Returns:
        清理后的payload
    """
    sanitized = {}
    xss_detected = False
    
    for key, value in payload.items():
        if isinstance(value, str):
            if detect_xss_attempt(value):
                xss_detected = True
                logger.warning(f"请求参数 '{key}' 包含潜在的XSS攻击尝试")
                # 清理值
                sanitized[key] = sanitize_user_input(value, log_attempts=False)
            else:
                sanitized[key] = value
        elif isinstance(value, (int, float, bool, type(None))):
            sanitized[key] = value
        elif isinstance(value, list):
            # 递归处理列表
            sanitized[key] = [
                sanitize_user_input(item, log_attempts=False) if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, dict):
            # 递归处理字典
            sanitized[key] = validate_and_sanitize_request(value)
        else:
            sanitized[key] = value
    
    if xss_detected:
        logger.warning("请求中包含XSS攻击尝试，已进行清理")
    
    return sanitized



