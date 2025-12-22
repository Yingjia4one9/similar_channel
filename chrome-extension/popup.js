const API_BASE = "http://127.0.0.1:8000";
let selectedTopics = [];
let currentResults = [];
let originalResults = []; // 保存原始结果，用于筛选
let currentSort = { field: "total_score", direction: "desc" };
let currentFilters = {
  topics: [],
  audience: [],
  minSubs: null,
  maxSubs: null
};
let isBdMode = false;  // BD模式状态
let bdSummary = null;  // BD模式摘要数据

// HTML转义函数（XSS防护）
function escapeHtml(text) {
  if (text == null || text === undefined) {
    return "";
  }
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

// Toast 提示
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
  
  container.appendChild(toast);
  
  // 自动移除
  setTimeout(() => {
    toast.classList.add('fade-out');
    toast.addEventListener('animationend', () => {
      toast.remove();
    });
  }, 3000);
}

// 渲染骨架屏
function renderSkeleton() {
  const container = document.getElementById("results");
  const count = 3; 
  
  let html = `
    <div class="loading-content">
      <div class="progress-container">
        <div class="progress-bar animated" id="progress-bar"></div>
      </div>
      <div class="loading-text" id="loading-text">正在分析相似频道，请稍候...</div>
    </div>
  `;
  
  for (let i = 0; i < count; i++) {
    html += `
      <div class="skeleton-card">
        <div class="skeleton-header">
          <div class="skeleton-avatar skeleton"></div>
          <div class="skeleton-info">
            <div class="skeleton-title skeleton"></div>
            <div class="skeleton-meta skeleton"></div>
          </div>
        </div>
        <div class="skeleton-tags skeleton"></div>
        <div class="skeleton-metrics">
          <div class="skeleton-metric skeleton"></div>
          <div class="skeleton-metric skeleton"></div>
          <div class="skeleton-metric skeleton"></div>
        </div>
      </div>
    `;
  }
  
  container.innerHTML = html;
}

document.getElementById("bd-mode").addEventListener("change", (e) => {
  isBdMode = e.target.checked;
  const toggle = document.getElementById("bd-mode-toggle");
  const bdSortOption = document.querySelector('.bd-sort-option');
  
  if (isBdMode) {
    toggle.classList.add("active");
    // 显示BD总分排序选项
    if (bdSortOption) bdSortOption.style.display = "flex";
    // BD模式下默认按BD总分排序
    currentSort = { field: "bd_total_score", direction: "desc" };
    // 更新排序选中状态
    document.querySelectorAll("#sort-menu .dropdown-item").forEach(i => i.classList.remove("selected"));
    if (bdSortOption) bdSortOption.classList.add("selected");
  } else {
    toggle.classList.remove("active");
    // 隐藏BD总分排序选项
    if (bdSortOption) {
      bdSortOption.style.display = "none";
      bdSortOption.classList.remove("selected");
    }
    // 普通模式按总评分排序
    currentSort = { field: "total_score", direction: "desc" };
    // 更新排序选中状态
    document.querySelectorAll("#sort-menu .dropdown-item").forEach(i => i.classList.remove("selected"));
    document.querySelector("#sort-menu .dropdown-item[data-sort='total_score']")?.classList.add("selected");
  }
  
  // 如果有结果，重新应用排序
  if (originalResults.length > 0) {
    applyFiltersAndSort();
  }
});

// Topic 按钮选择
document.querySelectorAll(".topic-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const topic = btn.dataset.topic;
    if (btn.classList.contains("selected")) {
      btn.classList.remove("selected");
      selectedTopics = selectedTopics.filter(t => t !== topic);
    } else {
      if (selectedTopics.length < 3) {
        btn.classList.add("selected");
        selectedTopics.push(topic);
      } else {
        showToast("最多只能选择 3 个主题", "info");
      }
    }
  });
});

function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(1) + "K";
  return num.toString();
}

function getInitials(name) {
  return name.split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
}

function renderResults(data) {
  const base = data.base_channel;
  const list = data.similar_channels || [];
  bdSummary = data.bd_summary || null;  // 保存BD摘要
  originalResults = list; // 保存原始结果
  currentResults = [...list]; // 创建副本用于排序和筛选
  applyFiltersAndSort();
  const container = document.getElementById("results");
  const countEl = document.getElementById("results-count");

  if (!currentResults.length) {
    container.innerHTML = '<div class="empty-state"><p>没有找到符合条件的相似频道</p></div>';
    countEl.textContent = "0 个结果";
    return;
  }

  // BD模式下显示摘要卡片
  let bdSummaryHtml = "";
  if (bdSummary) {
    bdSummaryHtml = `
      <div class="bd-summary-card">
        <div class="bd-summary-title">🎯 BD模式分析结果</div>
        <div class="bd-summary-stats">
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.high_priority || 0}</div>
            <div class="bd-summary-stat-label">🔥 高优先</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.medium_priority || 0}</div>
            <div class="bd-summary-stat-label">⚡ 中优先</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.with_email || 0}</div>
            <div class="bd-summary-stat-label">📧 有邮箱</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.with_competitor_collab || 0}</div>
            <div class="bd-summary-stat-label">🏢 有竞品</div>
          </div>
        </div>
      </div>
    `;
  }

  countEl.textContent = `Found ${currentResults.length} results (共 ${originalResults.length} 个)`;

  container.innerHTML = bdSummaryHtml + currentResults.map((ch, idx) => {
    const topics = ch.topics || [];
    const audience = ch.audience || [];
    const emails = ch.emails || [];
    const url = `https://www.youtube.com/channel/${ch.channelId}`;
    const thumbnails = ch.thumbnails || {};
    const thumbnailUrl = thumbnails.medium?.url || thumbnails.default?.url || "";

    const topicTags = topics.map(t => 
      `<span class="tag tag-topic">${t}</span>`
    ).join("");
    const audTags = audience.map(a => 
      `<span class="tag tag-audience">${a}</span>`
    ).join("");

    // BD模式专属数据
    const bdPriority = ch.bd_priority || "";
    const bdTotalScore = ch.bd_total_score || 0;
    const bdMetrics = ch.bd_metrics || {};
    const bdBreakdown = ch.bd_breakdown || {};
    const bdRecommendation = ch.bd_recommendation || {};
    const competitorDetection = bdMetrics.competitor_detection || {};
    
    // BD优先级徽章
    const priorityBadgeMap = {
      high: '<span class="bd-priority-badge bd-priority-high">🔥 高</span>',
      medium: '<span class="bd-priority-badge bd-priority-medium">⚡ 中</span>',
      low: '<span class="bd-priority-badge bd-priority-low">📌 低</span>',
      skip: '<span class="bd-priority-badge bd-priority-skip">⏭️ 不建议</span>',
    };
    const priorityBadge = bdPriority ? (priorityBadgeMap[bdPriority] || "") : "";

    // BD评分区域HTML
    let bdMetricsHtml = "";
    if (bdSummary && bdPriority) {
      const competitors = competitorDetection.competitors || [];
      const reasons = bdRecommendation.reasons || [];
      const concerns = bdRecommendation.concerns || [];
      
      bdMetricsHtml = `
        <div class="bd-metrics-section">
          <div class="bd-metrics-title">🎯 BD评分</div>
          <div class="bd-metrics-grid">
            <div class="bd-metric">
              <div class="bd-metric-value">${(bdTotalScore * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">总分</div>
            </div>
            <div class="bd-metric">
              <div class="bd-metric-value">${((bdBreakdown.contract_focus_score || 0) * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">合约</div>
            </div>
            <div class="bd-metric">
              <div class="bd-metric-value">${((bdBreakdown.commercialization_score || 0) * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">商业化</div>
            </div>
          </div>
          ${competitors.length > 0 ? `
            <div class="competitor-tags">
              <span style="font-size:10px;color:#92400e;">已合作: </span>
              ${competitors.map(c => `<span class="competitor-tag">${c}</span>`).join("")}
            </div>
          ` : ""}
          ${(reasons.length > 0 || concerns.length > 0) ? `
            <div class="bd-recommendation">
              ${reasons.length > 0 ? `<div class="bd-recommendation-reasons">✅ ${reasons.slice(0, 2).join(" · ")}</div>` : ""}
              ${concerns.length > 0 ? `<div class="bd-recommendation-concerns">⚠️ ${concerns.slice(0, 1).join(" · ")}</div>` : ""}
            </div>
          ` : ""}
        </div>
      `;
    }

    return `
      <div class="channel-card">
        <div class="channel-header">
          <div class="channel-avatar">
            ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${ch.title}" />` : getInitials(ch.title)}
          </div>
          <div class="channel-info">
            <div class="channel-name">
              <a href="${url}" target="_blank">${ch.title}</a>
              ${priorityBadge}
            </div>
            <div class="channel-handle">@${ch.channelId.substring(0, 12)}...</div>
            <div class="channel-meta">
              ${ch.country ? `<span class="country-flag">${ch.country === "US" ? "🇺🇸" : ch.country === "PK" ? "🇵🇰" : ch.country === "GB" ? "🇬🇧" : "🌍"}</span>` : ""}
              <span>${ch.country || ""}</span>
            </div>
          </div>
        </div>
        
        <div class="tags-section">
          <div class="tags-label">Topics</div>
          <div class="tags">${topicTags || "<span class='tag tag-topic'>-</span>"}</div>
        </div>
        
        <div class="tags-section">
          <div class="tags-label">Audience</div>
          <div class="tags">${audTags || "<span class='tag tag-audience'>-</span>"}</div>
        </div>

        <div class="metrics-grid">
          <div class="metric">
            <div class="metric-label">
              Subs
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">${formatNumber(ch.subscriberCount || 0)}</div>
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
            <div class="metric-value">${formatNumber(ch.viewCount || 0)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              Likes
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">${ch.avg_likes ? formatNumber(ch.avg_likes) : "-"}</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              E.R.
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">${ch.engagement_rate ? ch.engagement_rate.toFixed(1) + "%" : "-"}</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              V.R.
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">${ch.view_rate ? ch.view_rate.toFixed(1) + "%" : "-"}</div>
          </div>
        </div>

        ${bdMetricsHtml}

        ${(ch.recent_videos && ch.recent_videos.length > 0) ? `
          <div class="tags-section">
            <div class="tags-label">最近视频</div>
            <div class="video-thumbnails">
              ${ch.recent_videos.slice(0, 5).map(video => {
                const thumbnails = video.thumbnails || {};
                const thumbnailUrl = thumbnails.medium?.url || thumbnails.default?.url || "";
                const videoUrl = video.videoId ? `https://www.youtube.com/watch?v=${video.videoId}` : "#";
                return `
                  <a href="${videoUrl}" target="_blank" class="video-thumbnail" title="${video.title || ""}">
                    ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${video.title || ""}" />` : ""}
                  </a>
                `;
              }).join("")}
            </div>
          </div>
        ` : ""}

        ${emails.length > 0 ? `
          <div class="emails">
            <strong>Emails:</strong>
            <div class="emails-list">${emails.join(", ")}</div>
          </div>
        ` : ""}
      </div>
    `;
  }).join("");
}

document.getElementById("search-btn").addEventListener("click", async () => {
  const url = document.getElementById("channel-url").value.trim();
  if (!url) {
    showToast("请输入频道链接", "error");
    return;
  }

  const payload = {
    channel_url: url,
    max_results: Number(document.getElementById("max-results").value || 30),
    min_subscribers: document.getElementById("min-subs").value ? Number(document.getElementById("min-subs").value) : null,
    max_subscribers: document.getElementById("max-subs").value ? Number(document.getElementById("max-subs").value) : null,
    min_similarity: document.getElementById("min-sim").value ? Number(document.getElementById("min-sim").value) : null,
    bd_mode: isBdMode,  // BD模式参数
  };

  const resultsEl = document.getElementById("results");
  renderSkeleton();
  
  const progressBarEl = document.getElementById("progress-bar");
  const loadingTextEl = document.getElementById("loading-text");

  try {
    // 使用 fetch + ReadableStream 接收 SSE 流式进度更新
    const response = await fetch(`${API_BASE}/similar-channels/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showToast(`错误: ${errorData.detail || response.statusText}`, "error");
      resultsEl.innerHTML = "";
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
            
            if (data.type === "progress") {
              // 更新进度条
              const progress = data.progress || 0;
              if (progressBarEl) {
                progressBarEl.style.width = `${progress}%`;
                progressBarEl.classList.remove("animated");
              }
              if (loadingTextEl) {
                loadingTextEl.textContent = data.message || "正在处理...";
              }
            } else if (data.type === "result") {
              // 保存结果数据
              resultData = data.data;
            } else if (data.type === "error") {
              showToast(`错误: ${data.error}`, "error");
              resultsEl.innerHTML = "";
              return;
            } else if (data.type === "done") {
              // 完成，显示结果
              if (resultData) {
                renderResults(resultData);
                showToast("搜索完成", "success");
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
    console.error(err);
    showToast("请求失败，请确认后端已在 127.0.0.1:8000 运行。", "error");
    resultsEl.innerHTML = "";
  }
});

document.getElementById("export-btn").addEventListener("click", async () => {
  const url = document.getElementById("channel-url").value.trim();
  if (!url) {
    showToast("请先输入频道链接", "error");
    return;
  }

  const payload = {
    channel_url: url,
    max_results: Number(document.getElementById("max-results").value || 30),
    min_subscribers: document.getElementById("min-subs").value ? Number(document.getElementById("min-subs").value) : null,
    max_subscribers: document.getElementById("max-subs").value ? Number(document.getElementById("max-subs").value) : null,
    min_similarity: document.getElementById("min-sim").value ? Number(document.getElementById("min-sim").value) : null,
    bd_mode: isBdMode,  // BD模式参数
  };

  try {
    const res = await fetch(`${API_BASE}/similar-channels/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      showToast(`导出失败: ${data.detail || res.statusText}`, "error");
      return;
    }
    const text = await res.text();
    const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
    const urlObj = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = urlObj;
    a.download = "similar_channels.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(urlObj);
    showToast("导出成功", "success");
  } catch (err) {
    console.error(err);
    showToast("导出失败，请确认后端已在 127.0.0.1:8000 运行。", "error");
  }
});

// 应用筛选和排序
function applyFiltersAndSort() {
  // 先应用筛选
  let filtered = originalResults.filter(ch => {
    // Topics 筛选
    if (currentFilters.topics.length > 0) {
      const chTopics = new Set(ch.topics || []);
      const hasAnyTopic = currentFilters.topics.some(t => chTopics.has(t));
      if (!hasAnyTopic) return false;
    }
    
    // Audience 筛选
    if (currentFilters.audience.length > 0) {
      const chAudience = new Set(ch.audience || []);
      const hasAnyAudience = currentFilters.audience.some(a => chAudience.has(a));
      if (!hasAnyAudience) return false;
    }
    
    // 订阅数筛选
    const subs = ch.subscriberCount || 0;
    if (currentFilters.minSubs !== null && subs < currentFilters.minSubs) return false;
    if (currentFilters.maxSubs !== null && subs > currentFilters.maxSubs) return false;
    
    return true;
  });
  
  // 再应用排序
  filtered.sort((a, b) => {
    const field = currentSort.field;
    const dir = currentSort.direction === "asc" ? 1 : -1;
    
    let aVal = a[field] || 0;
    let bVal = b[field] || 0;
    
    // 处理数字类型
    if (typeof aVal === "number" && typeof bVal === "number") {
      return (aVal - bVal) * dir;
    }
    
    // 处理字符串类型
    if (typeof aVal === "string" && typeof bVal === "string") {
      return aVal.localeCompare(bVal) * dir;
    }
    
    return 0;
  });
  
  currentResults = filtered;
  updateResultsDisplay();
}

// 更新结果显示
function updateResultsDisplay() {
  const container = document.getElementById("results");
  const countEl = document.getElementById("results-count");
  
  if (!currentResults.length) {
    container.innerHTML = '<div class="empty-state"><p>没有找到符合条件的相似频道</p></div>';
    countEl.textContent = `0 个结果 (共 ${originalResults.length} 个)`;
    return;
  }
  
  countEl.textContent = `Found ${currentResults.length} results (共 ${originalResults.length} 个)`;
  
  // BD模式下显示摘要卡片
  let bdSummaryHtml = "";
  if (bdSummary) {
    bdSummaryHtml = `
      <div class="bd-summary-card">
        <div class="bd-summary-title">🎯 BD模式分析结果</div>
        <div class="bd-summary-stats">
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.high_priority || 0}</div>
            <div class="bd-summary-stat-label">🔥 高优先</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.medium_priority || 0}</div>
            <div class="bd-summary-stat-label">⚡ 中优先</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.with_email || 0}</div>
            <div class="bd-summary-stat-label">📧 有邮箱</div>
          </div>
          <div class="bd-summary-stat">
            <div class="bd-summary-stat-value">${bdSummary.with_competitor_collab || 0}</div>
            <div class="bd-summary-stat-label">🏢 有竞品</div>
          </div>
        </div>
      </div>
    `;
  }
  
  container.innerHTML = bdSummaryHtml + currentResults.map((ch, idx) => {
    const topics = ch.topics || [];
    const audience = ch.audience || [];
    const emails = ch.emails || [];
    const url = `https://www.youtube.com/channel/${ch.channelId}`;
    const thumbnails = ch.thumbnails || {};
    const thumbnailUrl = thumbnails.medium?.url || thumbnails.default?.url || "";

    const topicTags = topics.map(t => 
      `<span class="tag tag-topic">${t}</span>`
    ).join("");
    const audTags = audience.map(a => 
      `<span class="tag tag-audience">${a}</span>`
    ).join("");

    // BD模式专属数据
    const bdPriority = ch.bd_priority || "";
    const bdTotalScore = ch.bd_total_score || 0;
    const bdMetrics = ch.bd_metrics || {};
    const bdBreakdown = ch.bd_breakdown || {};
    const bdRecommendation = ch.bd_recommendation || {};
    const competitorDetection = bdMetrics.competitor_detection || {};
    
    // BD优先级徽章
    const priorityBadgeMap = {
      high: '<span class="bd-priority-badge bd-priority-high">🔥 高</span>',
      medium: '<span class="bd-priority-badge bd-priority-medium">⚡ 中</span>',
      low: '<span class="bd-priority-badge bd-priority-low">📌 低</span>',
      skip: '<span class="bd-priority-badge bd-priority-skip">⏭️ 不建议</span>',
    };
    const priorityBadge = bdPriority ? (priorityBadgeMap[bdPriority] || "") : "";

    // BD评分区域HTML
    let bdMetricsHtml = "";
    if (bdSummary && bdPriority) {
      const competitors = competitorDetection.competitors || [];
      const reasons = bdRecommendation.reasons || [];
      const concerns = bdRecommendation.concerns || [];
      
      bdMetricsHtml = `
        <div class="bd-metrics-section">
          <div class="bd-metrics-title">🎯 BD评分</div>
          <div class="bd-metrics-grid">
            <div class="bd-metric">
              <div class="bd-metric-value">${(bdTotalScore * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">总分</div>
            </div>
            <div class="bd-metric">
              <div class="bd-metric-value">${((bdBreakdown.contract_focus_score || 0) * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">合约</div>
            </div>
            <div class="bd-metric">
              <div class="bd-metric-value">${((bdBreakdown.commercialization_score || 0) * 100).toFixed(0)}%</div>
              <div class="bd-metric-label">商业化</div>
            </div>
          </div>
          ${competitors.length > 0 ? `
            <div class="competitor-tags">
              <span style="font-size:10px;color:#92400e;">已合作: </span>
              ${competitors.map(c => `<span class="competitor-tag">${c}</span>`).join("")}
            </div>
          ` : ""}
          ${(reasons.length > 0 || concerns.length > 0) ? `
            <div class="bd-recommendation">
              ${reasons.length > 0 ? `<div class="bd-recommendation-reasons">✅ ${reasons.slice(0, 2).join(" · ")}</div>` : ""}
              ${concerns.length > 0 ? `<div class="bd-recommendation-concerns">⚠️ ${concerns.slice(0, 1).join(" · ")}</div>` : ""}
            </div>
          ` : ""}
        </div>
      `;
    }

    return `
      <div class="channel-card">
        <div class="channel-header">
          <div class="channel-avatar">
            ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${ch.title}" />` : getInitials(ch.title)}
          </div>
          <div class="channel-info">
            <div class="channel-name">
              <a href="${url}" target="_blank">${ch.title}</a>
              ${priorityBadge}
            </div>
            <div class="channel-handle">@${ch.channelId.substring(0, 12)}...</div>
            <div class="channel-meta">
              ${ch.country ? `<span class="country-flag">${ch.country === "US" ? "🇺🇸" : ch.country === "PK" ? "🇵🇰" : ch.country === "GB" ? "🇬🇧" : "🌍"}</span>` : ""}
              <span>${ch.country || ""}</span>
            </div>
          </div>
        </div>
        
        <div class="tags-section">
          <div class="tags-label">Topics</div>
          <div class="tags">${topicTags || "<span class='tag tag-topic'>-</span>"}</div>
        </div>
        
        <div class="tags-section">
          <div class="tags-label">Audience</div>
          <div class="tags">${audTags || "<span class='tag tag-audience'>-</span>"}</div>
        </div>

        <div class="metrics-grid">
          <div class="metric">
            <div class="metric-label">
              Subs
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">${formatNumber(ch.subscriberCount || 0)}</div>
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
            <div class="metric-value">${formatNumber(ch.viewCount || 0)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              Likes
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              E.R.
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">
              V.R.
              <span class="info-icon">i</span>
            </div>
            <div class="metric-value">-</div>
          </div>
        </div>

        ${(ch.recent_videos && ch.recent_videos.length > 0) ? `
          <div class="tags-section">
            <div class="tags-label">最近视频</div>
            <div class="video-thumbnails">
              ${ch.recent_videos.slice(0, 5).map(video => {
                const thumbnails = video.thumbnails || {};
                const thumbnailUrl = thumbnails.medium?.url || thumbnails.default?.url || "";
                const videoUrl = video.videoId ? `https://www.youtube.com/watch?v=${video.videoId}` : "#";
                return `
                  <a href="${videoUrl}" target="_blank" class="video-thumbnail" title="${video.title || ""}">
                    ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${video.title || ""}" />` : ""}
                  </a>
                `;
              }).join("")}
            </div>
          </div>
        ` : ""}

        ${emails.length > 0 ? `
          <div class="emails">
            <strong>Emails:</strong>
            <div class="emails-list">${emails.join(", ")}</div>
          </div>
        ` : ""}
      </div>
    `;
  }).join("");
}

// 初始化筛选标签
function initFilterTags() {
  if (originalResults.length === 0) return;
  
  // 收集所有 Topics 和 Audience
  const allTopics = new Set();
  const allAudience = new Set();
  
  originalResults.forEach(ch => {
    (ch.topics || []).forEach(t => allTopics.add(t));
    (ch.audience || []).forEach(a => allAudience.add(a));
  });
  
  // 渲染 Topics 标签
  const topicsContainer = document.getElementById("filter-topics");
  topicsContainer.innerHTML = Array.from(allTopics).sort().map(topic => 
    `<span class="filter-tag" data-topic="${topic}">${topic}</span>`
  ).join("");
  
  // 渲染 Audience 标签
  const audienceContainer = document.getElementById("filter-audience");
  audienceContainer.innerHTML = Array.from(allAudience).sort().map(aud => 
    `<span class="filter-tag" data-audience="${aud}">${aud}</span>`
  ).join("");
  
  // 绑定点击事件
  topicsContainer.querySelectorAll(".filter-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      tag.classList.toggle("selected");
      updateFilterTopics();
    });
  });
  
  audienceContainer.querySelectorAll(".filter-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      tag.classList.toggle("selected");
      updateFilterAudience();
    });
  });
}

function updateFilterTopics() {
  const selected = Array.from(document.querySelectorAll("#filter-topics .filter-tag.selected"))
    .map(tag => tag.dataset.topic);
  currentFilters.topics = selected;
}

function updateFilterAudience() {
  const selected = Array.from(document.querySelectorAll("#filter-audience .filter-tag.selected"))
    .map(tag => tag.dataset.audience);
  currentFilters.audience = selected;
}

// Sort 功能
document.getElementById("sort-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  const menu = document.getElementById("sort-menu");
  menu.classList.toggle("active");
  
  // 点击外部关闭
  setTimeout(() => {
    document.addEventListener("click", function closeMenu() {
      menu.classList.remove("active");
      document.removeEventListener("click", closeMenu);
    });
  }, 0);
});

// 排序选项点击
document.querySelectorAll("#sort-menu .dropdown-item").forEach(item => {
  item.addEventListener("click", (e) => {
    e.stopPropagation();
    const field = item.dataset.sort;
    const currentDir = item.dataset.dir;
    
    // 如果点击的是当前选中的项，切换排序方向
    if (currentSort.field === field) {
      const newDir = currentDir === "asc" ? "desc" : "asc";
      item.dataset.dir = newDir;
      currentSort.direction = newDir;
      item.querySelector(".sort-direction").textContent = newDir === "asc" ? "↑" : "↓";
    } else {
      // 取消其他项的选中状态
      document.querySelectorAll("#sort-menu .dropdown-item").forEach(i => {
        i.classList.remove("selected");
      });
      // 选中当前项
      item.classList.add("selected");
      currentSort.field = field;
      currentSort.direction = currentDir;
      item.querySelector(".sort-direction").textContent = currentDir === "asc" ? "↑" : "↓";
    }
    
    applyFiltersAndSort();
    document.getElementById("sort-menu").classList.remove("active");
  });
});

// 默认选中总评分排序
document.querySelector("#sort-menu .dropdown-item[data-sort='total_score']").classList.add("selected");

// Filter 功能
document.getElementById("filter-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  const menu = document.getElementById("filter-menu");
  menu.classList.toggle("active");
  
  // 如果打开筛选面板，初始化标签
  if (menu.classList.contains("active") && originalResults.length > 0) {
    initFilterTags();
  }
  
  // 点击外部关闭
  setTimeout(() => {
    document.addEventListener("click", function closeMenu() {
      menu.classList.remove("active");
      document.removeEventListener("click", closeMenu);
    });
  }, 0);
});

// 应用筛选
document.getElementById("apply-filter").addEventListener("click", () => {
  currentFilters.minSubs = document.getElementById("filter-min-subs").value 
    ? Number(document.getElementById("filter-min-subs").value) : null;
  currentFilters.maxSubs = document.getElementById("filter-max-subs").value 
    ? Number(document.getElementById("filter-max-subs").value) : null;
  
  applyFiltersAndSort();
  document.getElementById("filter-menu").classList.remove("active");
});

// 重置筛选
document.getElementById("reset-filter").addEventListener("click", () => {
  currentFilters = {
    topics: [],
    audience: [],
    minSubs: null,
    maxSubs: null
  };
  
  // 清除 UI 状态
  document.querySelectorAll(".filter-tag").forEach(tag => tag.classList.remove("selected"));
  document.getElementById("filter-min-subs").value = "";
  document.getElementById("filter-max-subs").value = "";
  
  applyFiltersAndSort();
  document.getElementById("filter-menu").classList.remove("active");
});

