/**
 * 渲染模块
 * 处理频道卡片、标签、指标等的渲染
 */
import { formatNumber, getInitials, getCountryFlag, getThumbnailUrl } from "./utils.js";

/**
 * 渲染标签列表
 * @param {Array<string>} tags - 标签数组
 * @param {string} type - 标签类型 ('topic' 或 'audience')
 * @returns {string} HTML字符串
 */
function renderTags(tags, type) {
  if (!tags || tags.length === 0) {
    return `<span class='tag tag-${type}'>-</span>`;
  }
  return tags
    .map(t => `<span class="tag tag-${type}">${t}</span>`)
    .join("");
}

/**
 * 渲染视频缩略图列表
 * @param {Array<Object>} videos - 视频数组
 * @returns {string} HTML字符串
 */
function renderVideoThumbnails(videos) {
  if (!videos || videos.length === 0) return "";
  
  return videos.slice(0, 5)
    .map(video => {
      const thumbnailUrl = getThumbnailUrl(video.thumbnails);
      const videoUrl = video.videoId 
        ? `https://www.youtube.com/watch?v=${video.videoId}` 
        : "#";
      return `
        <a href="${videoUrl}" target="_blank" class="video-thumbnail" title="${video.title || ""}">
          ${thumbnailUrl ? `<img src="${thumbnailUrl}" alt="${video.title || ""}" />` : ""}
        </a>
      `;
    })
    .join("");
}

/**
 * 渲染单个频道卡片
 * @param {Object} ch - 频道对象
 * @returns {string} HTML字符串
 */
export function renderChannelCard(ch) {
  const topics = ch.topics || [];
  const audience = ch.audience || [];
  const emails = ch.emails || [];
  const url = `https://www.youtube.com/channel/${ch.channelId}`;
  const thumbnailUrl = getThumbnailUrl(ch.thumbnails);
  const country = ch.country || "";

  return `
    <div class="channel-card">
      <div class="channel-header">
        <div class="channel-avatar">
          ${thumbnailUrl 
            ? `<img src="${thumbnailUrl}" alt="${ch.title}" />` 
            : getInitials(ch.title)}
        </div>
        <div class="channel-info">
          <div class="channel-name">
            <a href="${url}" target="_blank">${ch.title}</a>
          </div>
          <div class="channel-handle">@${ch.channelId.substring(0, 12)}...</div>
          <div class="channel-meta">
            ${country ? `<span class="country-flag">${getCountryFlag(country)}</span>` : ""}
            <span>${country}</span>
          </div>
        </div>
      </div>
      
      <div class="tags-section">
        <div class="tags-label">Topics</div>
        <div class="tags">${renderTags(topics, "topic")}</div>
      </div>
      
      <div class="tags-section">
        <div class="tags-label">Audience</div>
        <div class="tags">${renderTags(audience, "audience")}</div>
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
            ${renderVideoThumbnails(ch.recent_videos)}
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
}

/**
 * 渲染结果列表
 * @param {Array<Object>} channels - 频道数组
 * @param {HTMLElement} container - 容器元素
 * @param {HTMLElement} countEl - 计数元素
 * @param {number} totalCount - 总数量
 */
export function renderResults(channels, container, countEl, totalCount) {
  if (!channels || channels.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>没有找到符合条件的相似频道</p></div>';
    if (countEl) countEl.textContent = `0 个结果 (共 ${totalCount || 0} 个)`;
    return;
  }

  if (countEl) {
    countEl.textContent = `Found ${channels.length} results (共 ${totalCount || channels.length} 个)`;
  }

  container.innerHTML = channels.map(ch => renderChannelCard(ch)).join("");
}

