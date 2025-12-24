/**
 * 公共前端代码模块（兼容版本，不使用ES6模块）
 * 包含所有共享的工具函数、渲染函数、API客户端和筛选排序逻辑
 */

(function(window) {
  'use strict';

  // ==================== 工具函数 ====================
  const Utils = {
    formatNumber: function(num) {
      if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
      if (num >= 1000) return (num / 1000).toFixed(1) + "K";
      return num.toString();
    },

    getInitials: function(name) {
      if (!name) return "??";
      return name.split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
    },

    getCountryFlag: function(country) {
      const flags = {
        US: "🇺🇸", PK: "🇵🇰", GB: "🇬🇧", CN: "🇨🇳", JP: "🇯🇵", KR: "🇰🇷",
        IN: "🇮🇳", BR: "🇧🇷", DE: "🇩🇪", FR: "🇫🇷", ES: "🇪🇸", IT: "🇮🇹",
        RU: "🇷🇺", CA: "🇨🇦", AU: "🇦🇺"
      };
      return flags[country] || "🌍";
    },

    getThumbnailUrl: function(thumbnails) {
      if (!thumbnails) return "";
      return thumbnails.medium?.url || thumbnails.default?.url || "";
    },

    // CP-y5-07：XSS防护 - HTML转义函数
    escapeHtml: function(text) {
      if (!text) return "";
      const div = document.createElement("div");
      div.textContent = text;
      return div.innerHTML;
    }
  };

  // ==================== 渲染函数 ====================
  const Renderer = {
    renderTags: function(tags, type) {
      if (!tags || tags.length === 0) {
        return `<span class='tag tag-${type}'>-</span>`;
      }
      // CP-y5-07：XSS防护 - 转义标签内容
      return tags.map(t => `<span class="tag tag-${type}">${Utils.escapeHtml(t)}</span>`).join("");
    },

    renderVideoThumbnails: function(videos) {
      if (!videos || videos.length === 0) return "";
      const self = this;
      return videos.slice(0, 5).map(video => {
        const thumbnailUrl = Utils.getThumbnailUrl(video.thumbnails);
        const videoUrl = video.videoId 
          ? `https://www.youtube.com/watch?v=${video.videoId}` 
          : "#";
        // CP-y5-07：XSS防护 - 转义视频标题
        const safeTitle = Utils.escapeHtml(video.title || "");
        return `
          <a href="${videoUrl}" target="_blank" class="video-thumbnail" title="${safeTitle}">
            ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${safeTitle}" />` : ""}
          </a>
        `;
      }).join("");
    },

    renderChannelCard: function(ch) {
      const topics = ch.topics || [];
      const audience = ch.audience || [];
      const emails = ch.emails || [];
      const url = `https://www.youtube.com/channel/${ch.channelId}`;
      const thumbnailUrl = Utils.getThumbnailUrl(ch.thumbnails);
      const country = ch.country || "";

      return `
        <div class="channel-card">
          <div class="channel-header">
            <div class="channel-avatar">
              ${thumbnailUrl 
                ? `<img src="${thumbnailUrl}" alt="${ch.title}" />` 
                : Utils.getInitials(ch.title)}
            </div>
            <div class="channel-info">
              <div class="channel-name">
                <a href="${url}" target="_blank">${Utils.escapeHtml(ch.title || "")}</a>
              </div>
              <div class="channel-handle">@${ch.channelId.substring(0, 12)}...</div>
              <div class="channel-meta">
                ${country ? `<span class="country-flag">${Utils.getCountryFlag(country)}</span>` : ""}
                <span>${Utils.escapeHtml(country || "")}</span>
              </div>
            </div>
          </div>
          
          <div class="tags-section">
            <div class="tags-label">Topics</div>
            <div class="tags">${this.renderTags(topics, "topic")}</div>
          </div>
          
          <div class="tags-section">
            <div class="tags-label">Audience</div>
            <div class="tags">${this.renderTags(audience, "audience")}</div>
          </div>

          <div class="metrics-grid">
            <div class="metric">
              <div class="metric-label">
                Subs
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${Utils.formatNumber(ch.subscriberCount || 0)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">
                Posts
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${ch.videoCount || 0}</div>
            </div>
            <div class="metric">
              <div class="metric-label">
                Views
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${Utils.formatNumber(ch.viewCount || 0)}</div>
            </div>
            <div class="metric">
              <div class="metric-label">
                Likes
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${ch.avg_likes ? Utils.formatNumber(ch.avg_likes) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">
                E.R.
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${ch.engagement_rate ? ch.engagement_rate + "%" : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">
                V.R.
                <span class="info-icon">i</span>
              </div>
              <div class="metric-value">${ch.view_rate ? ch.view_rate + "%" : "-"}</div>
            </div>
          </div>

          ${ch.recent_videos && ch.recent_videos.length > 0 ? `
            <div class="tags-section">
              <div class="tags-label">最近视频</div>
              <div class="video-thumbnails">
                ${this.renderVideoThumbnails(ch.recent_videos)}
              </div>
            </div>
          ` : ""}

          ${emails.length > 0 ? `
            <div class="emails">
              <strong>Emails:</strong>
              <div class="emails-list">${emails.map(e => Utils.escapeHtml(e)).join(", ")}</div>
            </div>
          ` : ""}
        </div>
      `;
    },

    renderResults: function(channels, container, countEl, totalCount) {
      if (!channels || channels.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>没有找到符合条件的相似频道</p></div>';
        if (countEl) countEl.textContent = `0 个结果 (共 ${totalCount || 0} 个)`;
        return;
      }

      if (countEl) {
        countEl.textContent = `Found ${channels.length} results (共 ${totalCount || channels.length} 个)`;
      }

      const self = this;
      container.innerHTML = channels.map(ch => self.renderChannelCard(ch)).join("");
    }
  };

  // ==================== API 客户端 ====================
  const APIClient = function(baseURL) {
    this.baseURL = baseURL || "http://127.0.0.1:8000";
  };

  APIClient.prototype.searchSimilarChannelsStream = function(params, callbacks) {
    const self = this;
    fetch(`${self.baseURL}/similar-channels/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.detail || response.statusText);
        }).catch(() => {
          throw new Error(response.statusText);
        });
      }
      return response;
    })
    .then(response => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let resultData = null;

      function processChunk() {
        reader.read().then(({ done, value }) => {
          if (done) return;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === "progress" && callbacks.onProgress) {
                  callbacks.onProgress(data.progress || 0, data.message || "正在处理...");
                } else if (data.type === "result") {
                  resultData = data.data;
                } else if (data.type === "error") {
                  if (callbacks.onError) callbacks.onError(data.error);
                  return;
                } else if (data.type === "done") {
                  if (resultData && callbacks.onResult) {
                    callbacks.onResult(resultData);
                  }
                  return;
                }
              } catch (e) {
                console.error("解析进度数据失败:", e, line);
              }
            }
          }
          processChunk();
        }).catch(err => {
          if (callbacks.onError) {
            callbacks.onError(`请求失败: ${err.message}`);
          }
        });
      }
      processChunk();
    })
    .catch(err => {
      console.error("API请求失败:", err);
      if (callbacks.onError) {
        callbacks.onError(`请求失败，请确认后端已在 ${self.baseURL} 运行。`);
      }
    });
  };

  APIClient.prototype.exportCSV = function(params) {
    const self = this;
    return fetch(`${self.baseURL}/similar-channels/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.detail || response.statusText);
        }).catch(() => {
          throw new Error(response.statusText);
        });
      }
      return response.text();
    });
  };

  APIClient.prototype.downloadCSV = function(params, filename) {
    filename = filename || "similar_channels.csv";
    const self = this;
    return this.exportCSV(params).then(csvText => {
      const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
      const urlObj = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = urlObj;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(urlObj);
    });
  };

  // ==================== 筛选和排序管理器 ====================
  const FilterSortManager = function() {
    this.currentSort = { field: "total_score", direction: "desc" };
    this.currentFilters = {
      topics: [],
      audience: [],
      minSubs: null,
      maxSubs: null,
    };
  };

  FilterSortManager.prototype.applyFiltersAndSort = function(originalResults) {
    const self = this;
    let filtered = originalResults.filter(ch => {
      if (self.currentFilters.topics.length > 0) {
        const chTopics = new Set(ch.topics || []);
        const hasAnyTopic = self.currentFilters.topics.some(t => chTopics.has(t));
        if (!hasAnyTopic) return false;
      }

      if (self.currentFilters.audience.length > 0) {
        const chAudience = new Set(ch.audience || []);
        const hasAnyAudience = self.currentFilters.audience.some(a => chAudience.has(a));
        if (!hasAnyAudience) return false;
      }

      const subs = ch.subscriberCount || 0;
      if (self.currentFilters.minSubs !== null && subs < self.currentFilters.minSubs) return false;
      if (self.currentFilters.maxSubs !== null && subs > self.currentFilters.maxSubs) return false;

      return true;
    });

    filtered.sort((a, b) => {
      const field = self.currentSort.field;
      const dir = self.currentSort.direction === "asc" ? 1 : -1;
      let aVal = a[field] || 0;
      let bVal = b[field] || 0;
      if (typeof aVal === "number" && typeof bVal === "number") {
        return (aVal - bVal) * dir;
      }
      if (typeof aVal === "string" && typeof bVal === "string") {
        return aVal.localeCompare(bVal) * dir;
      }
      return 0;
    });

    return filtered;
  };

  FilterSortManager.prototype.setSort = function(field, direction) {
    this.currentSort = { field, direction };
  };

  FilterSortManager.prototype.setFilters = function(filters) {
    this.currentFilters = Object.assign({}, this.currentFilters, filters);
  };

  FilterSortManager.prototype.resetFilters = function() {
    this.currentFilters = {
      topics: [],
      audience: [],
      minSubs: null,
      maxSubs: null,
    };
  };

  FilterSortManager.prototype.collectAvailableTags = function(results) {
    const allTopics = new Set();
    const allAudience = new Set();
    results.forEach(ch => {
      (ch.topics || []).forEach(t => allTopics.add(t));
      (ch.audience || []).forEach(a => allAudience.add(a));
    });
    return {
      allTopics: Array.from(allTopics).sort(),
      allAudience: Array.from(allAudience).sort(),
    };
  };

  // 导出到全局
  window.YTSimilarFrontend = {
    Utils: Utils,
    Renderer: Renderer,
    APIClient: APIClient,
    FilterSortManager: FilterSortManager
  };

})(window);

