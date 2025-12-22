# YouTube Similar Channels Finder - Chrome 扩展

这是一个 Chrome 浏览器扩展版本，用于查找相似的 YouTube 加密货币 KOL 频道。

## 两种使用模式

### 1. 侧边栏模式（推荐，类似 SimilarTube）⭐
- ✅ 在 YouTube 页面右侧显示侧边栏
- ✅ 自动检测当前频道并搜索
- ✅ 可以同时查看 YouTube 内容和搜索结果
- ✅ 工具栏按钮快速切换
- ✅ URL 变化监听，页面切换时自动更新搜索
- ✅ 状态保持，关闭浏览器后重新打开，侧边栏状态会恢复

### 2. 弹出窗口模式
- ✅ 点击扩展图标打开小窗口
- ✅ 独立窗口，不占用页面空间

**当前默认使用侧边栏模式**。如需使用弹出窗口模式，请修改 `manifest.json` 恢复 `action.default_popup` 配置。

## 功能特性

- 🔍 输入 YouTube 频道链接，查找相似的频道
- 📊 显示频道详细信息（订阅数、浏览量、视频数等）
- 🏷️ 自动识别频道主题和受众标签
- 📈 支持排序和筛选功能
- 📥 导出搜索结果为 CSV 文件
- ⚡ 实时进度显示
- 🎯 自动检测当前页面频道（侧边栏模式）

## 安装步骤

### 1. 准备后端服务

确保后端服务正在运行：

```bash
# 在项目根目录下
python main.py
# 或使用启动脚本
start.bat
```

后端服务默认运行在 `http://127.0.0.1:8000`

### 2. 准备扩展图标（可选但推荐）

#### 方法 1：使用 Python 脚本生成（推荐）

```bash
cd chrome-extension
pip install Pillow
python generate_icons.py
```

#### 方法 2：使用在线工具

访问以下网站生成图标：
- https://www.favicon-generator.org/
- https://favicon.io/
- https://realfavicongenerator.net/

#### 方法 3：手动创建

在 `chrome-extension/icons/` 目录下创建以下尺寸的图标：
- `icon16.png` (16x16 像素) - 工具栏图标
- `icon48.png` (48x48 像素) - 扩展管理页面
- `icon128.png` (128x128 像素) - Chrome 网上应用店

**图标设计建议**：
- 使用与扩展主题相关的颜色（建议使用紫色 #9333ea）
- 保持图标简洁，在小尺寸下也能清晰识别
- 可以使用 YouTube 相关的元素（如播放按钮、频道图标等）

如果暂时没有图标，可以创建简单的单色 PNG 图片作为占位符。

### 3. 加载扩展

1. 打开 Chrome 浏览器
2. 访问 `chrome://extensions/`
3. 开启右上角的"开发者模式"
4. 点击"加载已解压的扩展程序"
5. 选择 `chrome-extension` 文件夹
6. 扩展安装完成！

## 使用方法

### 侧边栏模式（推荐）

1. **访问 YouTube 频道页面**：
   - 打开任何 YouTube 频道页面（例如：`https://www.youtube.com/@Crypto621`）
   - 或访问频道的主页

2. **打开侧边栏**：
   - 点击 YouTube 工具栏上的 "🔍 Similar" 按钮
   - 侧边栏会从右侧滑出
   - 扩展会自动检测当前频道并开始搜索

3. **手动搜索**（可选）：
   - 在侧边栏中输入其他频道链接
   - 设置筛选条件（订阅数、相似度等）
   - 点击"Search"按钮

4. **查看结果**：
   - 搜索结果会显示在侧边栏中
   - 可以点击频道名称跳转到该频道
   - 点击"Export"可以导出 CSV 文件

5. **关闭侧边栏**：
   - 点击侧边栏右上角的 "×" 按钮
   - 或再次点击工具栏的 "🔍 Similar" 按钮

### 弹出窗口模式

如果使用弹出窗口模式：
1. 点击浏览器工具栏中的扩展图标
2. 在弹出的窗口中输入 YouTube 频道链接
3. 设置筛选条件并点击"Search"
4. 查看搜索结果

## 注意事项

- ⚠️ **必须确保后端服务正在运行**，扩展才能正常工作
- 扩展需要访问 `http://127.0.0.1:8000`，确保后端服务在该地址运行
- 如果遇到 CORS 错误，检查后端的 CORS 配置

## 故障排除

### 扩展无法连接到后端

1. 确认后端服务正在运行：访问 `http://127.0.0.1:8000/health` 应该返回 `{"status": "ok"}`
2. 检查防火墙设置，确保允许本地连接
3. 查看扩展的错误日志：在 `chrome://extensions/` 页面点击扩展的"错误"按钮

### 侧边栏不显示

1. 检查扩展是否已加载：访问 `chrome://extensions/` 确认扩展已启用
2. 检查控制台错误：按 F12 打开开发者工具查看错误
3. 确认后端服务运行：访问 `http://127.0.0.1:8000/health`
4. 重新加载扩展：在 `chrome://extensions/` 页面点击"重新加载"按钮，然后刷新 YouTube 页面

### 自动搜索不工作

1. 确认当前页面是频道页面（URL 包含 `/channel/` 或 `/@`）
2. 检查控制台是否有错误信息
3. 可以手动输入频道链接进行搜索

### 工具栏按钮不显示

1. 刷新 YouTube 页面
2. 检查扩展是否已启用
3. 如果仍不显示，可以右键点击扩展图标，选择"在 YouTube 上启用"

### 搜索结果为空

1. 确认输入的频道链接格式正确
2. 检查后端日志，查看是否有错误信息
3. 确认数据库索引已构建（运行 `build_channel_index.py`）

### 缺少图标错误

运行 `python generate_icons.py` 生成图标，或手动创建图标文件。

### 其他问题

1. **检查文件完整性**：确保以下文件存在
   - `manifest.json`
   - `content.js`
   - `sidebar.html`
   - `sidebar.js`
   - `sidebar.css`
   - `background.js`
   - `icons/icon16.png`
   - `icons/icon48.png`
   - `icons/icon128.png`

2. **清除扩展数据**：
   - 在 `chrome://extensions/` 页面
   - 找到扩展，点击"详细信息"
   - 点击"清除存储空间"
   - 重新加载扩展

3. **完全重新安装**：
   - 完全卸载扩展
   - 删除扩展文件夹
   - 重新下载/复制扩展文件
   - 重新加载扩展
   - 清除浏览器缓存

## 开发说明

### 文件结构

```
chrome-extension/
├── manifest.json      # 扩展配置文件
├── popup.html         # 弹出窗口 HTML
├── popup.css          # 弹出窗口样式
├── popup.js           # 弹出窗口逻辑
├── content.js         # 注入到 YouTube 页面的脚本
├── sidebar.html       # 侧边栏 HTML
├── sidebar.css        # 侧边栏样式
├── sidebar.js         # 侧边栏逻辑
├── background.js      # 后台脚本
├── generate_icons.py  # 图标生成脚本
├── icons/             # 图标文件目录
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── README.md          # 本文件
```

### 工作原理（侧边栏模式）

1. Content script (`content.js`) 在 YouTube 页面加载时运行
2. 创建工具栏按钮和侧边栏容器
3. 侧边栏使用 iframe 加载独立的 HTML 页面 (`sidebar.html`)
4. 通过 postMessage 在 content script 和 iframe 之间通信
5. 自动检测当前页面的频道 ID 并触发搜索

### 自定义配置

#### 修改 API 地址

如果需要修改后端 API 地址，编辑 `sidebar.js` 或 `popup.js` 文件中的 `API_BASE` 常量：

```javascript
const API_BASE = "http://127.0.0.1:8000";
```

#### 修改侧边栏宽度

编辑 `sidebar.css`：

```css
#yt-similar-sidebar {
  width: 400px; /* 修改为你想要的宽度 */
  right: -400px; /* 也要修改这个值 */
}
```

#### 修改自动搜索行为

编辑 `content.js` 中的 `extractChannelIdFromUrl()` 函数来自定义频道 ID 提取逻辑。

#### 自定义样式

修改 `popup.css` 或 `sidebar.css` 文件来自定义扩展的外观。

## 许可证

与主项目相同。
