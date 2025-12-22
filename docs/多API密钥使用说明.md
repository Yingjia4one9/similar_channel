# 多 API Key 使用说明

## 功能概述

系统现在支持为不同用途配置不同的 YouTube API Key，以提高总配额限制。

- **索引构建**：使用专用的 API Key（`YT_API_KEY_INDEX`）
- **实时搜索**：使用专用的 API Key（`YT_API_KEY_SEARCH`）
- **默认**：如果未配置专用 Key，则使用默认 Key（`YT_API_KEY`）

## 配置方式

### 方式1：环境变量（推荐）

#### Windows PowerShell
```powershell
# 索引构建专用
$env:YT_API_KEY_INDEX="your_index_api_key_here"

# 实时搜索专用
$env:YT_API_KEY_SEARCH="your_search_api_key_here"

# 默认（可选，如果未配置专用Key则使用此Key）
$env:YT_API_KEY="your_default_api_key_here"
```

#### Linux/Mac
```bash
# 索引构建专用
export YT_API_KEY_INDEX="your_index_api_key_here"

# 实时搜索专用
export YT_API_KEY_SEARCH="your_search_api_key_here"

# 默认（可选）
export YT_API_KEY="your_default_api_key_here"
```

### 方式2：文件配置

在项目根目录创建以下文件：

1. **`YouTube api key - index.txt`** - 索引构建专用 API Key
2. **`YouTube api key - search.txt`** - 实时搜索专用 API Key
3. **`YouTube api key.txt`** - 默认 API Key（如果未配置专用Key则使用）

每个文件只包含一行，即 API Key 字符串（不包含引号）。

## 配额优势

假设使用 2 个 API Key：

- **索引构建**：使用 Key1，配额 10,000 单位/天
- **实时搜索**：使用 Key2，配额 10,000 单位/天
- **总配额**：20,000 单位/天（提升 100%）

## 降级策略

如果未配置专用 API Key，系统会自动降级到默认 API Key：

1. 如果 `YT_API_KEY_INDEX` 或 `YouTube api key - index.txt` 不存在，索引构建会使用默认 Key
2. 如果 `YT_API_KEY_SEARCH` 或 `YouTube api key - search.txt` 不存在，实时搜索会使用默认 Key

这样可以确保向后兼容，即使只配置一个 API Key 也能正常工作。

## 使用场景

### 场景1：完全分离（推荐）

配置两个不同的 API Key：
- 索引构建使用 Key1
- 实时搜索使用 Key2

**优势**：
- 配额完全独立，互不影响
- 索引构建不会影响实时搜索的配额
- 总配额翻倍

### 场景2：部分分离

只配置一个专用 Key（如索引构建专用），另一个使用默认 Key。

**优势**：
- 索引构建有独立配额
- 实时搜索使用默认 Key
- 配置简单

### 场景3：统一使用（默认）

不配置专用 Key，所有操作都使用默认 Key。

**优势**：
- 配置最简单
- 向后兼容

## 注意事项

1. **配额追踪**：每个 API Key 的配额使用是独立追踪的
2. **错误处理**：如果某个 Key 配额用尽，系统会抛出 `YouTubeQuotaExceededError`
3. **安全性**：请妥善保管所有 API Key，不要提交到版本控制系统
4. **成本**：如果使用付费配额，多个 Key 的成本会相应增加

## 验证配置

配置完成后，可以通过以下方式验证：

1. **查看日志**：系统会在初始化时记录使用的 API Key 来源
   ```
   从环境变量加载索引构建专用API Key (YT_API_KEY_INDEX)
   从文件加载实时搜索专用API Key (YouTube api key - search.txt)
   ```

2. **测试运行**：
   - 运行 `build_channel_index.py` 应该使用索引构建专用 Key
   - 运行实时搜索应该使用实时搜索专用 Key

## 故障排查

### 问题：提示 "API Key 未配置"

**解决方案**：
- 检查环境变量是否正确设置
- 检查文件是否存在且包含有效的 API Key
- 确保 API Key 格式正确（不包含引号、换行符等）

### 问题：配额仍然不足

**可能原因**：
- 两个 Key 使用的是同一个 API Key（检查配置）
- API Key 配额本身已用尽
- 配额追踪可能不准确（检查配额使用情况）

### 问题：降级到默认 Key

**说明**：这是正常行为。如果未配置专用 Key，系统会自动使用默认 Key，确保向后兼容。

## 技术实现

- **配置加载**：`config.py` 中的 `load_api_key_for_index()` 和 `load_api_key_for_search()`
- **API 客户端**：`youtube_api.py` 中的 `get_api_client(use_for)` 支持不同用途
- **函数调用**：所有 API 调用函数都支持 `use_for` 参数

## 更新日志

- **v1.0** (2024): 初始实现，支持索引构建和实时搜索使用不同的 API Key

