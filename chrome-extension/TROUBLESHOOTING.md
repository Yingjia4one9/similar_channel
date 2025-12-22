# 故障排除指南

## 扩展无法打开/侧边栏不显示

### 1. 检查扩展是否已加载

1. 打开 Chrome，访问 `chrome://extensions/`
2. 确认扩展已启用（开关是打开的）
3. 检查是否有错误提示（红色错误图标）

### 2. 检查控制台错误

1. 访问 YouTube 页面
2. 按 `F12` 打开开发者工具
3. 查看 Console 标签页是否有错误信息
4. 常见错误：
   - `Failed to load resource` - 文件路径问题
   - `chrome.runtime is not defined` - manifest 配置问题
   - `CORS error` - 后端服务未运行

### 3. 重新加载扩展

1. 在 `chrome://extensions/` 页面
2. 找到扩展，点击"重新加载"按钮
3. 刷新 YouTube 页面

### 4. 检查文件完整性

确保以下文件存在：
- `manifest.json`
- `content.js`
- `sidebar.html`
- `sidebar.js`
- `sidebar.css`
- `background.js`
- `icons/icon16.png`
- `icons/icon48.png`
- `icons/icon128.png`

### 5. 检查后端服务

1. 确认后端服务正在运行：
   ```bash
   python main.py
   ```
2. 测试 API 是否可访问：
   - 打开浏览器，访问 `http://127.0.0.1:8000/health`
   - 应该返回 `{"status": "ok"}`

### 6. 检查 manifest.json

确保 manifest.json 格式正确，没有语法错误。可以使用在线 JSON 验证器检查。

### 7. 清除扩展数据

1. 在 `chrome://extensions/` 页面
2. 找到扩展，点击"详细信息"
3. 点击"清除存储空间"
4. 重新加载扩展

### 8. 常见问题

#### 问题：工具栏按钮不显示

**解决方案**：
- 刷新 YouTube 页面
- 检查 content.js 是否正确加载
- 查看控制台是否有错误

#### 问题：侧边栏显示但无法搜索

**解决方案**：
- 检查后端服务是否运行
- 查看控制台网络请求是否成功
- 确认 API_BASE 地址正确（`http://127.0.0.1:8000`）

#### 问题：iframe 无法加载

**解决方案**：
- 检查 `web_accessible_resources` 配置
- 确认 `sidebar.html` 和 `sidebar.js` 都在资源列表中
- 查看控制台是否有 CSP（内容安全策略）错误

### 9. 调试步骤

1. **检查 content script 是否运行**：
   - 在 YouTube 页面按 F12
   - 在 Console 中输入：`document.getElementById('yt-similar-sidebar')`
   - 如果返回 null，说明 content script 未运行

2. **检查侧边栏是否创建**：
   - 在 Console 中输入：`document.getElementById('yt-similar-toggle-btn')`
   - 如果返回 null，说明按钮未创建

3. **手动触发侧边栏**：
   - 在 Console 中输入：
   ```javascript
   chrome.runtime.sendMessage({action: 'toggle-sidebar'});
   ```

### 10. 如果仍然无法解决

1. 完全卸载扩展
2. 删除扩展文件夹
3. 重新下载/复制扩展文件
4. 重新加载扩展
5. 清除浏览器缓存

## 联系支持

如果以上方法都无法解决问题，请提供以下信息：
- Chrome 版本
- 扩展版本
- 错误信息截图
- 控制台错误日志

