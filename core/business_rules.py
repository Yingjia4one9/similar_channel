"""
业务规则模块
管理业务领域的规则和配置，如标签体系、权重配置等
与基础设施配置分离，支持动态加载和热更新
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TopicLabels:
    """Topics 标签配置"""
    labels: List[str] = field(default_factory=lambda: [
        # 主流币种
        "bitcoin",
        "ethereum",
        "solana",
        "bnb chain",
        "xrp ripple",
        "cardano",
        "polygon matic",
        "avalanche",
        "layer 2 solutions",
        
        # 币种类型
        "altcoins",
        "meme coins",
        "stablecoins",
        "privacy coins",
        
        # 交易类型（细分）
        "crypto trading",
        "spot trading",
        "futures trading",
        "options trading",
        "margin trading",
        "leverage trading",
        "crypto scalping",
        "swing trading",
        "position trading",
        "copy trading",
        "bot trading",
        "arbitrage trading",
        
        # 分析方法
        "technical analysis",
        "fundamental analysis",
        "on chain analysis",
        "sentiment analysis",
        "whale tracking",
        "liquidation analysis",
        
        # DeFi 细分
        "defi",
        "yield farming",
        "liquidity mining",
        "staking rewards",
        "lending protocols",
        "dex trading",
        "cross chain bridges",
        
        # NFT 细分
        "nft",
        "nft trading",
        "nft art",
        "gaming nft",
        
        # 其他领域
        "airdrop hunting",
        "crypto news",
        "crypto education",
        "market analysis",
        "ico ido launchpad",
        "crypto regulations",
        "crypto mining",
        "web3 development",
        "gamefi",
        "metaverse",
        
        # 内容类型
        "live trading streams",
        "trading signals",
        "portfolio management",
        "exchange reviews",
        "wallet tutorials",
    ])
    
    core_labels: List[str] = field(default_factory=lambda: [
        "bitcoin", "ethereum", "altcoins",
        "crypto trading", "futures trading", "spot trading",
        "technical analysis", "defi", "nft",
        "airdrop hunting", "crypto news", "crypto education",
    ])
    
    mutual_exclusion_groups: List[List[str]] = field(default_factory=lambda: [
        # 交易类型互斥组
        ["futures trading", "options trading", "margin trading", "leverage trading"],
        # 交易风格互斥组
        ["crypto scalping", "swing trading", "position trading"],
        # DeFi 细分互斥组
        ["yield farming", "liquidity mining", "staking rewards"],
        # NFT 细分互斥组
        ["nft", "nft trading", "nft art", "gaming nft"],
        # 分析方法互斥组
        ["technical analysis", "fundamental analysis"],
    ])


@dataclass
class AudienceLabels:
    """Audience 标签配置"""
    labels: List[str] = field(default_factory=lambda: [
        # 按经验级别
        "crypto beginners",
        "intermediate traders",
        "advanced traders",
        "professional traders",
        
        # 按交易风格
        "day traders",
        "swing traders",
        "scalpers",
        "long term investors",
        "hodlers",
        "active futures traders",
        "leverage traders",
        
        # 按资金规模
        "retail crypto traders",
        "high net worth traders",
        "whales",
        "institutional investors",
        
        # 按兴趣领域
        "defi power users",
        "nft collectors",
        "airdrop hunters",
        "yield farmers",
        "stakers",
        "miners",
        
        # 按内容偏好
        "crypto enthusiasts",
        "crypto learners",
        "trading signal followers",
        "technical analysts",
        "fundamentalists",
        
        # 按语言/地区
        "english speaking crypto",
        "spanish speaking crypto",
        "asian crypto market",
        
        # 其他
        "crypto content creators",
        "crypto developers",
        "crypto educators",
    ])
    
    core_labels: List[str] = field(default_factory=lambda: [
        "crypto beginners", "advanced traders", "day traders",
        "long term investors", "retail crypto traders", "whales",
        "defi power users", "nft collectors", "crypto enthusiasts",
    ])
    
    mutual_exclusion_groups: List[List[str]] = field(default_factory=lambda: [
        # 经验级别互斥组
        ["crypto beginners", "intermediate traders", "advanced traders", "professional traders"],
        # 交易风格互斥组
        ["day traders", "swing traders", "scalpers"],
        # 投资者类型互斥组
        ["long term investors", "hodlers"],
        # 资金规模互斥组
        ["retail crypto traders", "high net worth traders", "whales", "institutional investors"],
    ])


@dataclass
class BDRules:
    """BD模式业务规则（交易所BD寻找KOL）"""
    # 合约交易相关Topics标签（按优先级排序）
    contract_topics: List[str] = field(default_factory=lambda: [
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
    ])
    
    # 排除的Topics（这些通常不是合约交易受众）
    excluded_topics: List[str] = field(default_factory=lambda: [
        "defi",
        "nft",
        "airdrop hunting",
        "yield farming",
        "crypto education",
        "gamefi",
        "meme coins",
    ])
    
    # 受众标签（按交易活跃度优先级排序）
    audience_labels: List[str] = field(default_factory=lambda: [
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
    ])
    
    # 内容类型标签
    content_type_labels: List[str] = field(default_factory=lambda: [
        "daily market updates",      # 每日行情分析
        "live trading",              # 实盘直播
        "trading tutorials",         # 交易教程
        "trade setups",              # 交易布局分享
        "market analysis",           # 市场深度分析
        "trading signals",           # 交易信号
        "exchange reviews",          # 交易所测评
        "portfolio updates",         # 仓位分享
    ])
    
    # 竞品交易所检测配置
    competitor_exchanges: Dict[str, List[str]] = field(default_factory=lambda: {
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
    })
    
    # 商业化相关关键词
    collab_keywords: List[str] = field(default_factory=lambda: [
        "sponsor", "sponsored", "partnership", "partner",
        "合作", "赞助", "商务", "business",
        "referral", "affiliate", "返佣", "commission",
        "promo", "promotion", "code", "link",
    ])
    
    # 相似度权重配置
    similarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "contract_focus_score": 0.30,    # 合约内容聚焦度（最重要）
        "audience_quality_score": 0.20,  # 受众质量（真实交易者）
        "commercialization_score": 0.20, # 商业化潜力
        "engagement_rate_score": 0.15,   # 互动率
        "semantic_sim": 0.10,            # 语义相似度
        "scale_score": 0.05,             # 规模（订阅数）
    })
    
    # 合约聚焦度计算权重
    topic_weights: Dict[str, float] = field(default_factory=lambda: {
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
    })
    
    # 受众质量权重
    audience_weights: Dict[str, float] = field(default_factory=lambda: {
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
    })
    
    # 互动率评分阈值
    engagement_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 3.0,   # E.R. > 3% 优秀
        "good": 2.0,        # E.R. > 2% 良好
        "average": 1.0,     # E.R. > 1% 一般
        "poor": 0.5,        # E.R. < 0.5% 较差
    })
    
    # 优先级分类阈值
    priority_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.7,        # BD总分 > 0.7 高优先级
        "medium": 0.5,      # BD总分 > 0.5 中优先级
        "low": 0.3,         # BD总分 > 0.3 低优先级
    })
    
    # 自动保存配置
    auto_save: Dict[str, any] = field(default_factory=lambda: {
        "enabled": True,                    # 是否启用自动保存
        "save_all": False,                  # 是否保存所有频道（False时只保存高优先级）
        "min_priority": "medium",           # 最低保存优先级（high/medium/low）
        "save_base_channel": True,          # 是否保存基频道
    })


@dataclass
class SimilarityRules:
    """相似度计算规则"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "tag_score": 0.45,      # 标签相似度权重
        "semantic_sim": 0.40,    # 语义相似度权重
        "scale_score": 0.15,     # 订阅量级权重
    })


@dataclass
class TagInferenceRules:
    """标签推理规则"""
    max_topics: int = 10           # 最多 Topics 标签数
    max_audience: int = 8          # 最多 Audience 标签数
    core_threshold: float = 0.35   # 核心标签阈值（高置信度）
    extended_threshold: float = 0.25  # 扩展标签阈值（中置信度）
    enable_mutual_exclusion: bool = True  # 是否启用标签互斥
    threshold_base: float = 0.30   # 基础阈值
    threshold_ratio: float = 0.65  # 相对最高相似度的比例


@dataclass
class BusinessRules:
    """业务规则集合"""
    topics: TopicLabels = field(default_factory=TopicLabels)
    audience: AudienceLabels = field(default_factory=AudienceLabels)
    bd_rules: BDRules = field(default_factory=BDRules)
    similarity: SimilarityRules = field(default_factory=SimilarityRules)
    tag_inference: TagInferenceRules = field(default_factory=TagInferenceRules)
    
    @classmethod
    def load_default(cls) -> "BusinessRules":
        """加载默认业务规则"""
        return cls()
    
    @classmethod
    def load_from_dict(cls, data: Dict) -> "BusinessRules":
        """从字典加载业务规则（支持动态配置）"""
        # TODO: 实现从字典/JSON加载规则的功能
        # 目前先返回默认规则
        return cls.load_default()


# 全局业务规则实例（延迟初始化）
_business_rules: BusinessRules | None = None


def get_business_rules() -> BusinessRules:
    """
    获取业务规则实例（单例模式）
    
    Returns:
        BusinessRules 实例
    """
    global _business_rules
    if _business_rules is None:
        _business_rules = BusinessRules.load_default()
    return _business_rules


def reload_business_rules() -> BusinessRules:
    """
    重新加载业务规则（支持热更新）
    
    Returns:
        新的 BusinessRules 实例
    """
    global _business_rules
    _business_rules = BusinessRules.load_default()
    return _business_rules

