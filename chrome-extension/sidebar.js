const API_BASE = "http://127.0.0.1:8000";
let currentResults = [];
let originalResults = [];
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
    <div class="loading">
      <div class="progress-container">
        <div class="progress-bar animated" id="progress-bar"></div>
      </div>
      <div id="loading-text">正在分析相似频道，请稍候...</div>
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

// BD模式开关事件
document.getElementById("bd-mode").addEventListener("change", (e) => {
  isBdMode = e.target.checked;
  const toggle = document.getElementById("bd-mode-toggle");
  if (isBdMode) {
    toggle.classList.add("active");
  } else {
    toggle.classList.remove("active");
  }
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
  originalResults = list;
  currentResults = [...list];
  
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

  countEl.textContent = `找到 ${currentResults.length} 个结果 (共 ${originalResults.length} 个)`;

  container.innerHTML = bdSummaryHtml + currentResults.map((ch) => {
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
              <span style="font-size:9px;color:#92400e;">已合作: </span>
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
          </div>
        </div>
        
        <div class="tags">
          ${topicTags || "<span class='tag tag-topic'>-</span>"}
        </div>
        
        <div class="metrics">
          <div class="metric">订阅: ${formatNumber(ch.subscriberCount || 0)}</div>
          <div class="metric">视频: ${ch.videoCount || 0}</div>
          <div class="metric">浏览: ${formatNumber(ch.viewCount || 0)}</div>
          ${ch.engagement_rate ? `<div class="metric">E.R.: ${ch.engagement_rate.toFixed(1)}%</div>` : ""}
        </div>

        ${bdMetricsHtml}

        ${emails.length > 0 ? `
          <div style="margin-top: 8px; font-size: 10px; color: #9333ea;">
            📧 ${emails.join(", ")}
          </div>
        ` : ""}
      </div>
    `;
  }).join("");
}

async function performSearch(channelUrl) {
  if (!channelUrl) {
    showToast("请输入频道链接", "error");
    return;
  }

  const payload = {
    channel_url: channelUrl,
    max_results: Number(document.getElementById("max-results").value || 30),
    min_subscribers: document.getElementById("min-subs").value ? Number(document.getElementById("min-subs").value) : null,
    max_subscribers: document.getElementById("max-subs").value ? Number(document.getElementById("max-subs").value) : null,
    min_similarity: document.getElementById("min-sim").value ? Number(document.getElementById("min-sim").value) : null,
    bd_mode: isBdMode,  // BD模式参数
  };

  const resultsEl = document.getElementById("results");
  const countEl = document.getElementById("results-count");
  
  renderSkeleton();
  
  countEl.textContent = "搜索中...";
  const progressBarEl = document.getElementById("progress-bar");
  const loadingTextEl = document.getElementById("loading-text");

  try {
    const response = await fetch(`${API_BASE}/similar-channels/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      showToast(`错误: ${errorData.detail || response.statusText}`, "error");
      resultsEl.innerHTML = "";
      countEl.textContent = "搜索失败";
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
      buffer = lines.pop() || "";
      
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === "progress") {
              const progress = data.progress || 0;
              if (progressBarEl) {
                progressBarEl.style.width = `${progress}%`;
                progressBarEl.classList.remove("animated");
              }
              if (loadingTextEl) {
                loadingTextEl.textContent = data.message || "正在处理...";
              }
              countEl.textContent = data.message || "正在处理...";
            } else if (data.type === "result") {
              resultData = data.data;
            } else if (data.type === "error") {
              showToast(`错误: ${data.error}`, "error");
              resultsEl.innerHTML = "";
              countEl.textContent = "搜索失败";
              return;
            } else if (data.type === "done") {
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
    countEl.textContent = "连接失败";
  }
}

// 搜索按钮
document.getElementById("search-btn").addEventListener("click", async () => {
  const url = document.getElementById("channel-url").value.trim();
  await performSearch(url);
});

// 导出按钮
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

// 监听来自 content script 的消息
window.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'auto-search') {
    const channelUrl = event.data.channelUrl || `https://www.youtube.com/channel/${event.data.channelId}`;
    document.getElementById("channel-url").value = channelUrl;
    performSearch(channelUrl);
  }
});

// 通知父窗口已准备就绪
window.parent.postMessage({ type: 'sidebar-ready' }, '*');

