# YouTube 相似频道推荐后端

## 项目结构

项目已重新组织为以下结构：

```
yt-similar-backend/
├── app/                    # 应用入口
│   ├── __init__.py
│   └── main.py            # FastAPI 主应用
│
├── core/                   # 核心业务逻辑
│   ├── __init__.py
│   ├── youtube_api.py     # YouTube API 封装
│   ├── youtube_client.py  # YouTube 客户端
│   ├── channel_parser.py  # 频道 URL 解析
│   ├── channel_info.py    # 频道信息获取
│   ├── candidate_collection.py  # 候选频道收集
│   ├── similarity.py      # 相似度计算
│   ├── embedding.py       # 嵌入向量计算
│   └── bd_scoring.py      # BD 模式评分
│
├── infrastructure/        # 基础设施模块
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── logger.py          # 日志
│   ├── database.py        # 数据库操作
│   ├── cache.py           # 缓存
│   ├── result_cache.py    # 结果缓存
│   ├── quota_tracker.py   # 配额跟踪
│   ├── error_handler.py   # 错误处理
│   ├── validation.py      # 数据验证
│   ├── xss_protection.py  # XSS 防护
│   ├── utils.py           # 工具函数
│   └── metrics.py         # 指标统计
│
├── scripts/               # 工具脚本
│   ├── build_channel_index.py      # 构建频道索引
│   ├── clean_legacy_quota.py      # 清理旧配额记录
│   ├── check_module_dependencies.py  # 检查依赖
│   └── verify_p0_tasks.py          # 验证任务
│
├── data/                  # 数据文件
│   ├── channel_index.db   # 频道索引数据库
│   ├── quota_tracker.db   # 配额跟踪数据库
│   └── YouTube api key*.txt  # API 密钥文件
│
├── config/                # 配置文件
│   └── config.json.example # 配置示例
│
├── docs/                  # 文档
│   ├── 使用指南.md
│   ├── 模块架构文档.md
│   └── ...                # 其他文档
│
├── frontend/              # 前端文件
│   ├── css/
│   ├── js/
│   └── README.md
│
├── chrome-extension/      # Chrome 扩展
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── ...
│
├── requirements.txt       # Python 依赖
├── start.bat              # 启动脚本
└── README.md              # 本文件
```

## 启动服务

使用以下命令启动服务：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

或使用启动脚本：

```bash
start.bat
```

## 模块说明

### app/
FastAPI 应用入口，包含所有 API 路由和请求处理逻辑。

### core/
核心业务逻辑模块，包含 YouTube API 调用、频道信息处理、相似度计算等核心功能。

### infrastructure/
基础设施模块，提供配置管理、日志、数据库、缓存、配额跟踪等基础服务。

### scripts/
工具脚本，用于构建索引、清理数据、检查依赖等维护任务。

### data/
数据文件存储目录，包括数据库文件和 API 密钥文件（注意：API 密钥文件不应提交到版本控制）。

### config/
配置文件目录，包含配置示例文件。

### docs/
项目文档，包括使用指南、架构文档、质量检查清单等。

## 注意事项

1. **API 密钥文件**：`data/` 目录下的 API 密钥文件包含敏感信息，不应提交到版本控制系统。
2. **数据库文件**：数据库文件位于 `data/` 目录，请定期备份。
3. **配置文件**：复制 `config/config.json.example` 为 `config/config.json` 并根据需要修改。

