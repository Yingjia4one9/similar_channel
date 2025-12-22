"""
工具函数模块
包含文本处理、邮箱提取等通用功能
"""
import re
from typing import Any, Dict, List


def sanitize_sensitive_data(text: str, max_show: int = 4, mask_char: str = "*") -> str:
    """
    脱敏敏感数据（CP-y5-11：敏感数据脱敏）
    
    将敏感字符串（如API Key、文件路径）脱敏，只显示前后部分，中间用掩码字符替换。
    
    Args:
        text: 需要脱敏的文本
        max_show: 前后各显示多少个字符（默认4个）
        mask_char: 掩码字符（默认*）
    
    Returns:
        脱敏后的文本
    """
    if not text or not isinstance(text, str):
        return str(text) if text is not None else ""
    
    text = text.strip()
    length = len(text)
    
    # 如果文本太短，全部掩码
    if length <= max_show * 2:
        return mask_char * length
    
    # 显示前后各max_show个字符，中间用掩码
    prefix = text[:max_show]
    suffix = text[-max_show:]
    masked = mask_char * max(4, length - max_show * 2)
    
    return f"{prefix}{masked}{suffix}"


def sanitize_file_path(file_path: str) -> str:
    """
    脱敏文件路径中的敏感信息（CP-y5-11：敏感数据脱敏）
    
    移除或掩码文件路径中的用户名等敏感信息。
    
    Args:
        file_path: 文件路径
    
    Returns:
        脱敏后的路径
    """
    if not file_path or not isinstance(file_path, str):
        return str(file_path) if file_path is not None else ""
    
    # 在Windows路径中，掩码用户名部分（如 C:\Users\用户名\...）
    # 在Unix路径中，掩码home目录部分（如 /home/用户名/...）
    path = file_path.strip()
    
    # Windows路径：C:\Users\用户名\... -> C:\Users\***\...
    windows_pattern = r"(C:\\Users\\)([^\\]+)(\\.*)"
    if re.match(r"[A-Z]:\\", path, re.IGNORECASE):
        path = re.sub(windows_pattern, r"\1***\3", path, count=1)
    
    # Unix路径：/home/用户名/... -> /home/***/...
    unix_pattern = r"(/home/)([^/]+)(/.*)"
    if path.startswith("/home/"):
        path = re.sub(unix_pattern, r"\1***\2", path, count=1)
    
    # 掩码包含"api key"、"key"、"secret"、"password"、"token"等关键词的路径部分
    sensitive_keywords = ["api", "key", "secret", "password", "token", "credential"]
    path_lower = path.lower()
    for keyword in sensitive_keywords:
        if keyword in path_lower:
            # 找到包含关键词的路径段并掩码
            parts = path.split("/" if "/" in path else "\\")
            masked_parts = []
            for part in parts:
                if keyword in part.lower() and len(part) > 4:
                    masked_parts.append(sanitize_sensitive_data(part, max_show=2))
                else:
                    masked_parts.append(part)
            path = ("/" if "/" in path else "\\").join(masked_parts)
            break
    
    return path


def extract_emails_from_text(text: str) -> List[str]:
    """
    从一段文本中粗略提取邮箱地址（用于从频道简介 / 视频简介里找商务邮箱）。
    不保证 100% 精准，但对常见形式足够。
    
    Args:
        text: 要搜索的文本
    
    Returns:
        找到的邮箱地址列表（已去重）
    """
    if not text:
        return []
    # 非特别严格的邮箱正则，尽量兼顾常见写法
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    found = re.findall(pattern, text)
    # 去重、保持顺序
    seen = set()
    emails: List[str] = []
    for e in found:
        if e not in seen:
            seen.add(e)
            emails.append(e)
    return emails


def build_text_for_channel(info: Dict[str, Any]) -> str:
    """
    把频道的各种文本信息拼到一起，作为向量模型的输入。
    会包含：
    - 频道标题
    - 频道简介
    - 若存在 recent_videos 字段，则再加入最近视频的标题和简介
    
    Args:
        info: 频道信息字典
    
    Returns:
        拼接后的文本字符串
    """
    parts: List[str] = []
    title = info.get("title", "")
    desc = info.get("description", "")
    if title:
        parts.append(title)
    if desc:
        parts.append(desc)

    recent_videos = info.get("recent_videos") or []
    for v in recent_videos:
        vt = v.get("title", "")
        vd = v.get("description", "")
        if vt:
            parts.append(vt)
        if vd:
            parts.append(vd)

    return "\n\n".join(parts)

