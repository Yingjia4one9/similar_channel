# YouTube 相似频道推荐系统

一个基于 FastAPI 的 YouTube 相似频道推荐服务，支持语义相似度计算、多维度筛选、BD 模式评分等功能。

## 功能特性

- 🔍 **智能相似度匹配**：基于语义嵌入向量计算频道相似度
- 📊 **多维度筛选**：支持订阅数、语言、地区、相似度等筛选条件
- 🎯 **BD 模式**：交易所 BD 寻找 KOL 专用模式，计算合约聚焦度、商业化潜力等指标
- 📈 **配额管理**：完整的 YouTube API 配额跟踪和监控
- 🔄 **结果缓存**：智能缓存搜索结果，支持 CSV 导出
- ⚡ **流式搜索**：支持流式返回搜索结果，实时显示进度
- 🔐 **多 API Key 支持**：支持为不同用途（索引构建、实时搜索）配置不同的 API Key
- 🛡️ **安全防护**：XSS 防护、请求验证、错误处理等安全特性
- 🌐 **Chrome 扩展**：提供浏览器扩展，支持侧边栏和弹出窗口两种模式
- 📱 **Web 前端**：提供独立的 Web 前端界面

## 项目结构

```
yt-similar-backend/
├── app/                    # 应用入口
│   ├── __init__.py
│   └── main.py            # FastAPI 主应用，包含所有 API 路由
│
├── core/                   # 核心业务逻辑
│   ├── __init__.py
│   ├── youtube_api.py     # YouTube API 封装
│   ├── youtube_client.py  # YouTube 客户端主逻辑
│   ├── channel_parser.py  # 频道 URL 解析
│   ├── channel_info.py    # 频道信息获取
│   ├── candidate_collection.py  # 候选频道收集
│   ├── similarity.py      # 相似度计算
│   ├── embedding.py       # 嵌入向量计算
│   └── bd_scoring.py      # BD 模式评分
│
├── infrastructure/        # 基础设施模块
│   ├── __init__.py
│   ├── config.py          # 配置管理（环境变量、文件配置）
│   ├── logger.py          # 日志管理
│   ├── database.py        # SQLite 数据库操作（FAISS 支持）
│   ├── cache.py           # 内存缓存
│   ├── result_cache.py    # 结果缓存
│   ├── quota_tracker.py   # YouTube API 配额跟踪
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
│   ├── export_database.py         # 导出数据库
│   └── view_database.py           # 查看数据库内容
│
├── data/                  # 数据文件
│   ├── channel_index.db   # 频道索引数据库
│   ├── quota_tracker.db   # 配额跟踪数据库
│   ├── exports/           # CSV 导出目录
│   └── YouTube api key*.txt  # API 密钥文件（不应提交到版本控制）
│
├── config/                # 配置文件
│   └── config.json.example # 配置示例
│
├── docs/                  # 文档
│   ├── 模块架构文档.md     # 详细的模块架构说明
│   └── 多API密钥使用说明.md # 多 API Key 配置指南
│
├── frontend/              # Web 前端文件
│   ├── css/
│   │   └── styles.css     # 前端样式
│   ├── js/
│   │   ├── common.js      # 共享模块（兼容版本）
│   │   ├── utils.js       # 工具函数
│   │   ├── renderer.js    # 渲染函数
│   │   ├── api-client.js  # API 客户端
│   │   └── filter-sort.js # 筛选排序
│   └── README.md          # 前端说明文档
│
├── chrome-extension/      # Chrome 浏览器扩展
│   ├── manifest.json      # 扩展配置文件
│   ├── background.js      # 后台脚本
│   ├── content.js         # 内容脚本
│   ├── sidebar.html/js/css # 侧边栏模式
│   ├── popup.html/js/css  # 弹出窗口模式
│   └── README.md          # 扩展说明文档
│
├── requirements.txt       # Python 依赖
├── start.bat              # Windows 启动脚本
├── frontend.html          # Web 前端入口
└── README.md              # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

#### 方式 1：环境变量（推荐）

**Windows PowerShell:**
```powershell
# 默认 API Key（用于实时搜索）
$env:YT_API_KEY="your_api_key_here"

# 索引构建专用（可选）
$env:YT_API_KEY_INDEX="your_index_api_key_here"

# 实时搜索专用（可选）
$env:YT_API_KEY_SEARCH="your_search_api_key_here"
```

**Linux/Mac:**
```bash
export YT_API_KEY="your_api_key_here"
export YT_API_KEY_INDEX="your_index_api_key_here"  # 可选
export YT_API_KEY_SEARCH="your_search_api_key_here"  # 可选
```

#### 方式 2：文件配置

在 `data/` 目录下创建以下文件：
- `YouTube api key.txt` - 默认 API Key
- `YouTube api key - index.txt` - 索引构建专用（可选）
- `YouTube api key - search.txt` - 实时搜索专用（可选）

每个文件只包含一行，即 API Key 字符串（不包含引号）。

> 💡 **提示**：支持多 API Key 可以显著提高配额限制。详见 [多API密钥使用说明](docs/多API密钥使用说明.md)

### 3. 构建频道索引（可选但推荐）

构建本地频道索引可以加速搜索并减少 API 调用：

```bash
python scripts/build_channel_index.py
```

### 4. 启动服务

**方式 1：使用启动脚本（Windows）**
```bash
start.bat
```

**方式 2：使用 uvicorn**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务启动后，默认运行在 `http://127.0.0.1:8000`

### 5. 访问 Web 前端

打开浏览器访问：`http://127.0.0.1:8000` 或直接打开 `frontend.html`

## API 文档

### 核心 API

#### 1. 搜索相似频道

**POST** `/similar-channels`

搜索与指定频道相似的频道列表。

**请求体：**
```json
{
  "channel_url": "https://www.youtube.com/@example",
  "max_results": 20,
  "min_subscribers": 1000,
  "max_subscribers": 1000000,
  "preferred_language": "en",
  "preferred_region": "US",
  "min_similarity": 0.5,
  "bd_mode": false
}
```

**响应：**
```json
{
  "result_id": "xxx-xxx-xxx",
  "base_channel": {...},
  "similar_channels": [...],
  "quota_exceeded_channels": [],
  "failed_channels": [],
  "bd_summary": null
}
```

#### 2. 流式搜索相似频道

**POST** `/similar-channels/stream`

流式返回搜索结果，实时显示进度。

#### 3. 导出搜索结果

**POST** `/similar-channels/export`

导出搜索结果为 CSV 格式。

**请求体：**
```json
{
  "result_id": "xxx-xxx-xxx"  // 优先使用 result_id，不消耗 API 配额
  // 或使用搜索参数（会消耗 API 配额）
}
```

### 配额管理 API

#### 4. 查询配额使用情况

**GET** `/quota/usage`

查询今天的配额使用情况。

#### 5. 按端点统计配额使用

**GET** `/quota/usage/by-endpoint?days=1`

按端点统计最近 N 天的配额使用情况。

#### 6. 查询配额状态

**GET** `/quota/status`

查询配额状态（包含是否耗尽、是否告警等）。

#### 7. 查询配额统计

**GET** `/quota/statistics?days=7`

查询详细的配额使用统计信息。

#### 8. 查询配额使用率

**GET** `/quota/usage-rate`

查询当前配额使用率（0-100）。

#### 9. 查询降级统计

**GET** `/quota/fallback-stats?days=1`

查询配额耗尽时的降级统计信息。

#### 10. 查询配额警告

**GET** `/quota/warnings`

查询配额警告列表。

#### 11. 确认配额警告

**POST** `/quota/warnings/{warning_id}/acknowledge`

确认并关闭配额警告。

### 缓存管理 API

#### 12. 查询缓存统计

**GET** `/cache/stats`

查询结果缓存统计信息。

### 健康检查

#### 13. 健康检查

**GET** `/health`

检查服务状态，返回 `{"status": "ok"}`。

## 使用场景

### 1. 查找相似频道

输入一个 YouTube 频道链接，系统会：
1. 分析该频道的主题、标签、描述等信息
2. 在本地索引或通过 API 搜索候选频道
3. 计算语义相似度和其他匹配指标
4. 返回相似度排序的频道列表

### 2. BD 模式（交易所 BD 寻找 KOL）

启用 `bd_mode=true` 后，系统会额外计算：
- 合约聚焦度
- 商业化潜力
- 受众匹配度
- 合作价值评分

适用于交易所 BD 团队寻找合适的 KOL 合作。

### 3. 批量筛选频道

支持多种筛选条件：
- **订阅数范围**：`min_subscribers` / `max_subscribers`
- **语言偏好**：`preferred_language`
- **地区偏好**：`preferred_region`
- **相似度阈值**：`min_similarity`

## Chrome 扩展

项目包含一个 Chrome 浏览器扩展，支持两种使用模式：

1. **侧边栏模式**（推荐）：在 YouTube 页面右侧显示侧边栏，自动检测当前频道并搜索
2. **弹出窗口模式**：点击扩展图标打开独立窗口

安装和使用说明详见：[chrome-extension/README.md](chrome-extension/README.md)

## 模块说明

### app/
FastAPI 应用入口，包含所有 API 路由和请求处理逻辑。

### core/
核心业务逻辑模块：
- **youtube_client.py**：主要的相似频道查找逻辑
- **youtube_api.py**：YouTube API 调用封装
- **embedding.py**：语义嵌入向量计算
- **similarity.py**：相似度计算算法
- **bd_scoring.py**：BD 模式评分算法

### infrastructure/
基础设施模块，提供：
- **config.py**：配置管理（支持环境变量和文件配置）
- **logger.py**：统一日志管理
- **database.py**：SQLite 数据库操作（支持 FAISS 向量搜索）
- **cache.py**：内存缓存（频道信息、视频列表等）
- **result_cache.py**：搜索结果缓存
- **quota_tracker.py**：YouTube API 配额跟踪和统计
- **validation.py**：数据验证
- **xss_protection.py**：XSS 攻击防护

详细架构说明详见：[docs/模块架构文档.md](docs/模块架构文档.md)

## 技术栈

- **后端框架**：FastAPI
- **数据库**：SQLite + FAISS（向量搜索）
- **机器学习**：sentence-transformers（语义嵌入）
- **API 客户端**：requests
- **前端**：原生 JavaScript + CSS
- **扩展**：Chrome Extension API

## 注意事项

1. **API 密钥安全**
   - `data/` 目录下的 API 密钥文件包含敏感信息，不应提交到版本控制系统
   - 建议使用环境变量或 `.gitignore` 排除密钥文件

2. **数据库备份**
   - 数据库文件位于 `data/` 目录，请定期备份
   - 可以使用 `scripts/export_database.py` 导出数据

3. **配额管理**
   - YouTube API 有每日配额限制（默认 10,000 单位/天）
   - 系统会自动跟踪配额使用情况
   - 配额用尽时，系统会尝试使用本地缓存数据

4. **配置管理**
   - 复制 `config/config.json.example` 为 `config/config.json` 并根据需要修改
   - 或使用环境变量配置（推荐用于生产环境）

5. **性能优化**
   - 建议先构建频道索引（`build_channel_index.py`）以加速搜索
   - 结果缓存可以避免重复计算
   - 使用流式搜索可以获得更好的用户体验

## 故障排查

### 服务无法启动

1. 检查 Python 版本（建议 Python 3.8+）
2. 确认所有依赖已安装：`pip install -r requirements.txt`
3. 检查端口 8000 是否被占用

### API 调用失败

1. 检查 API Key 是否正确配置
2. 查看日志文件了解详细错误信息
3. 检查配额是否用尽：访问 `/quota/status` 端点

### 搜索结果为空

1. 确认输入的频道链接格式正确
2. 检查本地索引是否已构建（如使用本地索引）
3. 查看后端日志了解详细错误信息

### 配额相关问题

1. 查看配额使用情况：访问 `/quota/usage` 端点
2. 配置多个 API Key 以提高配额限制
3. 清理旧配额记录：运行 `scripts/clean_legacy_quota.py`

更多问题请查看各模块的文档或提交 Issue。

## 开发说明

### 代码结构

项目采用分层架构设计：
- **基础设施层**：配置、日志、数据库、缓存等基础服务
- **数据访问层**：数据库操作、API 调用
- **业务逻辑层**：核心业务处理
- **应用层**：API 路由和请求处理

详细架构说明请参考：[docs/模块架构文档.md](docs/模块架构文档.md)

### 扩展开发

1. **添加新的筛选条件**：修改 `app/main.py` 中的 `SimilarChannelRequest` 模型
2. **自定义相似度算法**：修改 `core/similarity.py`
3. **添加新的 API 端点**：在 `app/main.py` 中添加路由

## 许可证

[根据项目实际情况添加许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！
