"""
配置管理模块
统一管理 Topics/Audience 标签、API 配置等
支持环境变量、配置文件、多环境配置
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """应用配置类"""
    
    # 配置文件路径
    CONFIG_FILE_ENV_VAR = "YT_CONFIG_FILE"
    DEFAULT_CONFIG_FILE = "config.json"
    
    # YouTube API Key 配置
    API_KEY_ENV_VAR = "YT_API_KEY"
    API_KEY_FILE = "YouTube api key.txt"
    
    # 多 API Key 配置（用于不同用途，提高配额）
    API_KEY_INDEX_ENV_VAR = "YT_API_KEY_INDEX"  # 索引构建专用
    API_KEY_SEARCH_ENV_VAR = "YT_API_KEY_SEARCH"  # 实时搜索专用
    API_KEY_INDEX_FILE = "YouTube api key - index.txt"
    API_KEY_SEARCH_FILE = "YouTube api key - search.txt"
    
    # 环境配置
    ENV_ENV_VAR = "YT_ENV"
    DEFAULT_ENV = "development"  # development, production, testing
    
    # 配置缓存
    _config_cache: Optional[Dict[str, Any]] = None
    _env: Optional[str] = None
    
    # 嵌入模型配置
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
    
    # Topics 标签（加密货币领域）
    TOPIC_LABELS: List[str] = [
        "bitcoin",
        "ethereum",
        "altcoins",
        "crypto trading",
        "crypto scalping",
        "futures trading",
        "spot trading",
        "on chain analysis",
        "technical analysis",
        "crypto news",
        "defi",
        "nft",
        "airdrop hunting",
        "yield farming",
        "crypto education",
    ]
    
    # Audience 标签
    AUDIENCE_LABELS: List[str] = [
        "crypto beginners",
        "retail crypto traders",
        "advanced traders",
        "day traders",
        "long term investors",
        "whales",
        "institutional investors",
        "defi power users",
        "nft collectors",
        "crypto enthusiasts",
        "crypto educators",
        "crypto content creators",
    ]
    
    # ==================== BD模式配置（交易所BD寻找KOL） ====================
    
    # BD模式：合约交易相关Topics标签（按优先级排序）
    BD_CONTRACT_TOPICS: List[str] = [
        # 核心目标（高优先级）- 合约/杠杆交易
        "futures trading",
        "leverage trading",
        "perpetual contracts",
        "margin trading",
        "crypto scalping",
        "liquidation analysis",
        # 强相关（中优先级）
        "technical analysis",
        "crypto trading signals",
        "whale watching",
        "funding rate",
        "copy trading",
        "day trading",
        # 弱相关（低优先级）
        "spot trading",
        "bitcoin price prediction",
        "altcoin trading",
        "crypto news",
    ]
    
    # BD模式：排除的Topics（这些通常不是合约交易受众）
    BD_EXCLUDED_TOPICS: List[str] = [
        "defi",
        "nft",
        "airdrop hunting",
        "yield farming",
        "crypto education",
        "gamefi",
        "meme coins",
    ]
    
    # BD模式：受众标签（按交易活跃度优先级排序）
    BD_AUDIENCE_LABELS: List[str] = [
        # 高价值目标受众（活跃交易者）
        "active futures traders",
        "leverage traders",
        "scalpers",
        "day traders",
        "swing traders",
        # 中等价值（有交易习惯）
        "retail crypto traders",
        "advanced traders",
        "crypto enthusiasts",
        # 低价值（可能只是观众）
        "crypto beginners",
        "crypto learners",
        "long term investors",
    ]
    
    # BD模式：内容类型标签
    BD_CONTENT_TYPE_LABELS: List[str] = [
        "daily market updates",      # 每日行情分析
        "live trading",              # 实盘直播
        "trading tutorials",         # 交易教程
        "trade setups",              # 交易布局分享
        "market analysis",           # 市场深度分析
        "trading signals",           # 交易信号
        "exchange reviews",          # 交易所测评
        "portfolio updates",         # 仓位分享
    ]
    
    # BD模式：竞品交易所检测配置
    COMPETITOR_EXCHANGES: Dict[str, List[str]] = {
        "binance": [
            "binance.com/referral", "binance.com/register", "binance.com/activity",
            "accounts.binance.com", "binance.com/en/register",
        ],
        "bybit": [
            "bybit.com/register", "partner.bybit.com", "bybit.com/referral",
            "bybit.com/en-US/register", "bybit.com/invite",
        ],
        "okx": [
            "okx.com/join", "okx.com/referral", "okx.com/account/register",
            "okx.com/cn/join", "okex.com",
        ],
        "bitget": [
            "bitget.com/referral", "partner.bitget.com", "bitget.com/register",
            "bitget.com/en/referral",
        ],
        "gate": [
            "gate.io/referral", "gate.io/signup", "gate.io/ref",
        ],
        "kucoin": [
            "kucoin.com/ucenter/signup", "kucoin.com/referral", "kucoin.com/land",
        ],
        "mexc": [
            "mexc.com/register", "mexc.com/referral",
        ],
        "htx": [
            "htx.com/invite", "htx.com/register", "huobi.com",
        ],
        "bingx": [
            "bingx.com/invite", "bingx.com/register", "bingx.com/referral",
        ],
        "phemex": [
            "phemex.com/register", "phemex.com/referral",
        ],
    }
    
    # BD模式：商业化相关关键词
    BD_COLLAB_KEYWORDS: List[str] = [
        "sponsor", "sponsored", "partnership", "partner",
        "合作", "赞助", "商务", "business",
        "referral", "affiliate", "返佣", "commission",
        "promo", "promotion", "code", "link",
    ]
    
    # BD模式：相似度权重配置
    BD_SIMILARITY_WEIGHTS: Dict[str, float] = {
        "contract_focus_score": 0.30,    # 合约内容聚焦度（最重要）
        "audience_quality_score": 0.20,  # 受众质量（真实交易者）
        "commercialization_score": 0.20, # 商业化潜力
        "engagement_rate_score": 0.15,   # 互动率
        "semantic_sim": 0.10,            # 语义相似度
        "scale_score": 0.05,             # 规模（订阅数）
    }
    
    # BD模式：合约聚焦度计算权重
    BD_TOPIC_WEIGHTS: Dict[str, float] = {
        # 核心合约相关（权重1.0）
        "futures trading": 1.0,
        "leverage trading": 1.0,
        "perpetual contracts": 1.0,
        "margin trading": 1.0,
        "crypto scalping": 1.0,
        "liquidation analysis": 1.0,
        # 强相关（权重0.7）
        "technical analysis": 0.7,
        "crypto trading signals": 0.7,
        "whale watching": 0.7,
        "funding rate": 0.7,
        "copy trading": 0.7,
        "day trading": 0.7,
        # 弱相关（权重0.3）
        "spot trading": 0.3,
        "bitcoin price prediction": 0.3,
        "altcoin trading": 0.3,
        "crypto news": 0.3,
        # 不相关/负面（权重-0.3）
        "defi": -0.3,
        "nft": -0.3,
        "airdrop hunting": -0.3,
        "yield farming": -0.3,
        "crypto education": -0.2,
    }
    
    # BD模式：受众质量权重
    BD_AUDIENCE_WEIGHTS: Dict[str, float] = {
        # 高价值（权重1.0）
        "active futures traders": 1.0,
        "leverage traders": 1.0,
        "scalpers": 1.0,
        "day traders": 0.9,
        "swing traders": 0.8,
        # 中等价值（权重0.5）
        "retail crypto traders": 0.5,
        "advanced traders": 0.5,
        "crypto enthusiasts": 0.4,
        # 低价值（权重0.1-0.2）
        "crypto beginners": 0.2,
        "crypto learners": 0.1,
        "long term investors": 0.3,
    }
    
    # BD模式：互动率评分阈值
    BD_ENGAGEMENT_THRESHOLDS: Dict[str, float] = {
        "excellent": 3.0,   # E.R. > 3% 优秀
        "good": 2.0,        # E.R. > 2% 良好
        "average": 1.0,     # E.R. > 1% 一般
        "poor": 0.5,        # E.R. < 0.5% 较差
    }
    
    # BD模式：优先级分类阈值
    BD_PRIORITY_THRESHOLDS: Dict[str, float] = {
        "high": 0.7,        # BD总分 > 0.7 高优先级
        "medium": 0.5,      # BD总分 > 0.5 中优先级
        "low": 0.3,         # BD总分 > 0.3 低优先级
    }
    
    # 相似度计算权重
    SIMILARITY_WEIGHTS = {
        "tag_score": 0.45,      # 标签相似度权重
        "semantic_sim": 0.40,    # 语义相似度权重
        "scale_score": 0.15,     # 订阅量级权重
    }
    
    # 标签推理阈值
    TAG_THRESHOLD_BASE = 0.35    # 基础阈值
    TAG_THRESHOLD_RATIO = 0.7    # 相对最高相似度的比例
    
    # 候选频道收集配置
    CANDIDATE_COLLECTION = {
        "local_index_max": 120,
        "related_videos_per_video": 30,
        "related_videos_limit": 100,
        "title_search_limit": 80,
        "topic_search_limit": 50,
        "audience_search_limit": 30,
        "max_candidates": 200,
    }
    
    # 频道信息获取配置
    CHANNEL_INFO = {
        "recent_videos_count": 5,
        "recent_videos_for_similarity": 3,
        "stats_videos_count": 10,
        "batch_size": 50,  # 批量获取频道信息的批次大小（YouTube API限制：最多50个）
    }
    
    # 订阅数筛选配置
    SUBSCRIBER_FILTER = {
        "min_dynamic_ratio": 0.01,  # 动态最小订阅数 = 基频道订阅数 * 0.01
        "min_absolute": 1000,       # 绝对最小订阅数（至少1k以上）
    }
    
    # 相似度评分配置
    SCALE_SCORE = {
        "max_diff": 3.0,  # log10 差值超过 3（即相差 1000 倍）时得分为 0
    }
    
    # 并发处理配置
    CONCURRENT_PROCESSING = {
        "search_workers": 5,      # 主题/受众搜索的线程池大小
        "video_fetch_workers": 10, # 视频信息获取的线程池大小
        "index_build_workers": 5,  # 索引构建的线程池大小
    }
    
    # CORS配置
    CORS_ALLOW_ORIGINS_ENV = "CORS_ALLOW_ORIGINS"
    CORS_ALLOW_ORIGINS_DEFAULT = ["*"]  # 开发环境默认允许所有来源
    
    # 缓存TTL配置（秒）
    CACHE_TTL = {
        "channel_info": 7200,      # 频道信息缓存2小时
        "channel_videos": 1800,    # 视频列表缓存30分钟
    }
    
    # 向量批量计算配置
    EMBEDDING_BATCH_SIZE = 32  # 向量编码批量大小（根据内存调整，32-128）
    
    # API请求超时配置（秒）
    API_TIMEOUT = 10  # YouTube API请求超时时间（建议5-15秒）
    DB_TIMEOUT = 30.0  # 数据库连接超时时间（秒）
    
    # 数据库批量操作配置
    DB_BATCH_SIZE = 500  # 数据库批量插入/更新的批次大小
    
    # 配额速率限制配置
    QUOTA_RATE_LIMIT = {
        "enabled": True,              # 是否启用配额速率限制
        "threshold": 0.8,              # 配额使用率阈值（80%时开始限流）
        "strict_threshold": 0.95,      # 严格限流阈值（95%时启用严格限流）
        "reduction_rate": 0.5,         # 限流时的请求速率降低比例（50%）
        "min_delay_seconds": 0.1,      # 最小延迟时间（秒）
        "max_delay_seconds": 5.0,      # 最大延迟时间（秒）
    }
    
    @staticmethod
    def get_thread_pool_size(key: str, default: int) -> int:
        """
        获取线程池大小，支持通过环境变量覆盖
        
        Args:
            key: 配置键名（如 "search_workers"）
            default: 默认值
        
        Returns:
            线程池大小
        """
        env_key = f"THREAD_POOL_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                pass
        return default
    
    @staticmethod
    def get_cors_allow_origins() -> list[str]:
        """
        获取CORS允许的来源列表
        
        Returns:
            CORS允许的来源列表
        """
        origins_str = os.getenv(Config.CORS_ALLOW_ORIGINS_ENV, "")
        if origins_str:
            # 从环境变量读取，支持逗号分隔的多个来源
            return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
        return Config.CORS_ALLOW_ORIGINS_DEFAULT
    
    @classmethod
    def get_env(cls) -> str:
        """
        获取当前环境
        
        Returns:
            环境名称（development, production, testing）
        """
        if cls._env is None:
            cls._env = os.getenv(cls.ENV_ENV_VAR, cls.DEFAULT_ENV).lower()
        return cls._env
    
    @classmethod
    def load_config_file(cls) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if cls._config_cache is not None:
            return cls._config_cache
        
        config_file = os.getenv(cls.CONFIG_FILE_ENV_VAR, cls.DEFAULT_CONFIG_FILE)
        # 如果配置文件路径不是绝对路径，则从 config 文件夹查找
        if not os.path.isabs(config_file):
            config_path = Path(os.path.dirname(__file__)) / ".." / "config" / config_file
        else:
            config_path = Path(config_file)
        
        # 如果配置文件不存在，返回空字典
        if not config_path.exists():
            logger = cls._get_logger()
            # 脱敏文件路径（CP-y5-11：敏感数据脱敏）
            from infrastructure.utils import sanitize_file_path
            safe_path = sanitize_file_path(str(config_path))
            logger.debug(f"配置文件不存在: {safe_path}，使用默认配置")
            cls._config_cache = {}
            return cls._config_cache
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # 验证配置文件不包含敏感信息（安全检查）
            cls._validate_config_security(config_data)
            
            # 根据环境选择配置
            env = cls.get_env()
            if env in config_data:
                env_config = config_data[env]
            elif "default" in config_data:
                env_config = config_data["default"]
            else:
                env_config = config_data
            
            cls._config_cache = env_config
            logger = cls._get_logger()
            logger.debug(f"成功加载配置文件（环境: {env}）")
            return cls._config_cache
        except json.JSONDecodeError as e:
            logger = cls._get_logger()
            from infrastructure.utils import sanitize_file_path
            safe_path = sanitize_file_path(str(config_path))
            logger.warning(f"配置文件格式错误: {e}（路径: {safe_path}），使用默认配置")
            cls._config_cache = {}
            return cls._config_cache
        except Exception as e:
            logger = cls._get_logger()
            from infrastructure.utils import sanitize_file_path
            safe_path = sanitize_file_path(str(config_path))
            logger.warning(f"加载配置文件失败: {type(e).__name__}（路径: {safe_path}），使用默认配置")
            cls._config_cache = {}
            return cls._config_cache
    
    @classmethod
    def get_config_value(cls, key: str, default: Any = None, env_override: Optional[str] = None) -> Any:
        """
        获取配置值，优先级：环境变量 > 配置文件 > 默认值
        
        Args:
            key: 配置键名（支持点号分隔的嵌套键，如 "database.host"）
            default: 默认值
            env_override: 环境变量名（如果为None，则使用 key.upper()）
        
        Returns:
            配置值
        """
        # 1. 优先从环境变量读取
        env_key = env_override or key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            # 尝试转换为适当类型
            if isinstance(default, bool):
                return env_value.lower() in ("true", "1", "yes", "on")
            elif isinstance(default, int):
                try:
                    return int(env_value)
                except ValueError:
                    pass
            elif isinstance(default, float):
                try:
                    return float(env_value)
                except ValueError:
                    pass
            return env_value
        
        # 2. 从配置文件读取
        config_data = cls.load_config_file()
        if config_data:
            keys = key.split(".")
            value = config_data
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    value = None
                    break
            if value is not None:
                return value
        
        # 3. 返回默认值
        return default
    
    @classmethod
    def _get_logger(cls):
        """延迟导入logger，避免循环依赖"""
        from infrastructure.logger import get_logger
        return get_logger()
    
    @classmethod
    def _validate_config_security(cls, config_data: Dict[str, Any]) -> None:
        """
        验证配置文件安全性，确保不包含敏感信息。
        
        Args:
            config_data: 配置数据字典
            
        Raises:
            ValueError: 如果发现敏感信息
        """
        # 敏感关键词列表
        sensitive_keys = [
            "api_key", "apikey", "api-key", "secret", "password", "passwd",
            "token", "access_token", "refresh_token", "credential",
            "private_key", "privatekey", "private-key"
        ]
        
        def check_dict(d: Any, path: str = "") -> None:
            """递归检查字典中的敏感信息"""
            if not isinstance(d, dict):
                return
            
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key
                key_lower = str(key).lower()
                
                # 检查键名是否包含敏感关键词
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    logger = cls._get_logger()
                    logger.warning(
                        f"配置文件包含可能的敏感配置项: {current_path}。"
                        f"建议使用环境变量而不是配置文件存储敏感信息。"
                    )
                
                # 递归检查嵌套字典
                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            check_dict(item, f"{current_path}[{i}]")
        
        check_dict(config_data)
    
    @staticmethod
    def _load_api_key_from_source(env_var: str, file_name: str, purpose: str = "") -> str | None:
        """
        从环境变量或文件加载 API Key（内部辅助方法）
        
        Args:
            env_var: 环境变量名
            file_name: 文件名
            purpose: 用途描述（用于日志）
        
        Returns:
            API Key 字符串，如果未找到则返回 None
        """
        logger = Config._get_logger()
        
        # 优先从环境变量读取
        key = os.getenv(env_var)
        if key:
            logger.debug(f"从环境变量加载{purpose}API Key ({env_var})")
            if len(key.strip()) < 10:
                logger.warning("API Key长度异常，可能无效")
            return key.strip()
        
        # 从文件中读取（从 data 文件夹读取）
        key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", file_name))
        if os.path.exists(key_path):
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if key:
                    logger.debug(f"从文件加载{purpose}API Key ({file_name})")
                    if len(key) < 10:
                        logger.warning("API Key长度异常，可能无效")
                    return key
            except OSError as e:
                from infrastructure.utils import sanitize_file_path
                safe_path = sanitize_file_path(str(key_path))
                logger.debug(f"读取{purpose}API Key文件失败: {e} (路径: {safe_path})")
            except Exception as e:
                logger.debug(f"加载{purpose}API Key时发生错误: {type(e).__name__}")
        
        return None
    
    @staticmethod
    def load_api_key() -> str:
        """
        加载默认 YouTube API Key。
        优先使用环境变量，其次从文件读取。
        
        Returns:
            API Key 字符串
            
        Raises:
            ValueError: 如果无法找到 API Key
        """
        logger = Config._get_logger()
        
        key = Config._load_api_key_from_source(
            Config.API_KEY_ENV_VAR, 
            Config.API_KEY_FILE,
            "默认"
        )
        
        if key:
            return key
        
        # API Key未找到
        logger.error(
            f"API Key未配置。请设置环境变量 {Config.API_KEY_ENV_VAR} "
            f"或提供 '{Config.API_KEY_FILE}' 文件。"
        )
        raise ValueError(
            f"YouTube API Key 未配置，请设置环境变量 {Config.API_KEY_ENV_VAR} "
            f"或提供 '{Config.API_KEY_FILE}' 文件。"
        )
    
    @staticmethod
    def load_api_key_for_index() -> str:
        """
        加载索引构建专用的 YouTube API Key。
        优先使用专用配置，如果未配置则降级到默认 API Key。
        
        Returns:
            API Key 字符串
            
        Raises:
            ValueError: 如果无法找到任何 API Key
        """
        logger = Config._get_logger()
        
        # 尝试加载专用 key
        key = Config._load_api_key_from_source(
            Config.API_KEY_INDEX_ENV_VAR,
            Config.API_KEY_INDEX_FILE,
            "索引构建专用"
        )
        
        if key:
            return key
        
        # 降级到默认 key
        logger.debug("未找到索引构建专用API Key，使用默认API Key")
        return Config.load_api_key()
    
    @staticmethod
    def load_api_key_for_search() -> str:
        """
        加载实时搜索专用的 YouTube API Key。
        优先使用专用配置，如果未配置则降级到默认 API Key。
        
        Returns:
            API Key 字符串
            
        Raises:
            ValueError: 如果无法找到任何 API Key
        """
        logger = Config._get_logger()
        
        # 尝试加载专用 key
        key = Config._load_api_key_from_source(
            Config.API_KEY_SEARCH_ENV_VAR,
            Config.API_KEY_SEARCH_FILE,
            "实时搜索专用"
        )
        
        if key:
            return key
        
        # 降级到默认 key
        logger.debug("未找到实时搜索专用API Key，使用默认API Key")
        return Config.load_api_key()

