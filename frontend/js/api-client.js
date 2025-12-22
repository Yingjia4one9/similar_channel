/**
 * API 客户端模块
 * 处理与后端的API通信
 */
export class APIClient {
  constructor(baseURL = "http://127.0.0.1:8000") {
    this.baseURL = baseURL;
  }

  /**
   * 搜索相似频道（流式）
   * @param {Object} params - 搜索参数
   * @param {Function} onProgress - 进度回调 (progress, message)
   * @param {Function} onResult - 结果回调 (data)
   * @param {Function} onError - 错误回调 (error)
   */
  async searchSimilarChannelsStream(params, { onProgress, onResult, onError }) {
    try {
      const response = await fetch(`${this.baseURL}/similar-channels/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const error = errorData.detail || response.statusText;
        if (onError) onError(error);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let resultData = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // 保留最后一个不完整的行

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "progress" && onProgress) {
                onProgress(data.progress || 0, data.message || "正在处理...");
              } else if (data.type === "result") {
                resultData = data.data;
              } else if (data.type === "error") {
                if (onError) onError(data.error);
                return;
              } else if (data.type === "done") {
                if (resultData && onResult) {
                  onResult(resultData);
                }
                return;
              }
            } catch (e) {
              console.error("解析进度数据失败:", e, line);
            }
          }
        }
      }
    } catch (err) {
      console.error("API请求失败:", err);
      if (onError) {
        onError(`请求失败，请确认后端已在 ${this.baseURL} 运行。`);
      }
    }
  }

  /**
   * 导出CSV
   * @param {Object} params - 搜索参数
   * @returns {Promise<string>} CSV文本
   */
  async exportCSV(params) {
    const response = await fetch(`${this.baseURL}/similar-channels/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || response.statusText);
    }

    return await response.text();
  }

  /**
   * 下载CSV文件
   * @param {Object} params - 搜索参数
   * @param {string} filename - 文件名
   */
  async downloadCSV(params, filename = "similar_channels.csv") {
    try {
      const csvText = await this.exportCSV(params);
      const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
      const urlObj = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = urlObj;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(urlObj);
    } catch (err) {
      console.error("导出失败:", err);
      throw err;
    }
  }
}

