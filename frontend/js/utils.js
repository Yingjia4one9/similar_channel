/**
 * 工具函数模块
 * 提供格式化、文本处理等通用功能
 */

/**
 * 格式化数字（K/M 格式）
 * @param {number} num - 要格式化的数字
 * @returns {string} 格式化后的字符串
 */
export function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(1) + "K";
  return num.toString();
}

/**
 * 获取名称的首字母缩写
 * @param {string} name - 名称
 * @returns {string} 首字母缩写（最多2个字符）
 */
export function getInitials(name) {
  if (!name) return "??";
  return name
    .split(" ")
    .map(n => n[0])
    .join("")
    .substring(0, 2)
    .toUpperCase();
}

/**
 * 获取国家对应的国旗emoji
 * @param {string} country - 国家代码
 * @returns {string} 国旗emoji
 */
export function getCountryFlag(country) {
  const flags = {
    US: "🇺🇸",
    PK: "🇵🇰",
    GB: "🇬🇧",
    CN: "🇨🇳",
    JP: "🇯🇵",
    KR: "🇰🇷",
    IN: "🇮🇳",
    BR: "🇧🇷",
    DE: "🇩🇪",
    FR: "🇫🇷",
    ES: "🇪🇸",
    IT: "🇮🇹",
    RU: "🇷🇺",
    CA: "🇨🇦",
    AU: "🇦🇺",
  };
  return flags[country] || "🌍";
}

/**
 * 获取缩略图URL
 * @param {Object} thumbnails - 缩略图对象
 * @returns {string} 缩略图URL
 */
export function getThumbnailUrl(thumbnails) {
  if (!thumbnails) return "";
  return thumbnails.medium?.url || thumbnails.default?.url || "";
}

/**
 * HTML转义函数（CP-y5-07：XSS防护）
 * 转义HTML特殊字符，防止XSS攻击
 * @param {string} text - 要转义的文本
 * @returns {string} 转义后的文本
 */
export function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

