/**
 * 筛选和排序模块
 * 处理结果的筛选、排序逻辑
 */

/**
 * 筛选和排序管理器
 */
export class FilterSortManager {
  constructor() {
    this.currentSort = { field: "total_score", direction: "desc" };
    this.currentFilters = {
      topics: [],
      audience: [],
      minSubs: null,
      maxSubs: null,
    };
  }

  /**
   * 应用筛选和排序
   * @param {Array<Object>} originalResults - 原始结果
   * @returns {Array<Object>} 筛选和排序后的结果
   */
  applyFiltersAndSort(originalResults) {
    // 先应用筛选
    let filtered = originalResults.filter(ch => {
      // Topics 筛选
      if (this.currentFilters.topics.length > 0) {
        const chTopics = new Set(ch.topics || []);
        const hasAnyTopic = this.currentFilters.topics.some(t => chTopics.has(t));
        if (!hasAnyTopic) return false;
      }

      // Audience 筛选
      if (this.currentFilters.audience.length > 0) {
        const chAudience = new Set(ch.audience || []);
        const hasAnyAudience = this.currentFilters.audience.some(a => chAudience.has(a));
        if (!hasAnyAudience) return false;
      }

      // 订阅数筛选
      const subs = ch.subscriberCount || 0;
      if (this.currentFilters.minSubs !== null && subs < this.currentFilters.minSubs) return false;
      if (this.currentFilters.maxSubs !== null && subs > this.currentFilters.maxSubs) return false;

      return true;
    });

    // 再应用排序
    filtered.sort((a, b) => {
      const field = this.currentSort.field;
      const dir = this.currentSort.direction === "asc" ? 1 : -1;

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

    return filtered;
  }

  /**
   * 设置排序
   * @param {string} field - 排序字段
   * @param {string} direction - 排序方向 ('asc' 或 'desc')
   */
  setSort(field, direction) {
    this.currentSort = { field, direction };
  }

  /**
   * 设置筛选条件
   * @param {Object} filters - 筛选条件对象
   */
  setFilters(filters) {
    this.currentFilters = { ...this.currentFilters, ...filters };
  }

  /**
   * 重置筛选条件
   */
  resetFilters() {
    this.currentFilters = {
      topics: [],
      audience: [],
      minSubs: null,
      maxSubs: null,
    };
  }

  /**
   * 收集所有可用的 Topics 和 Audience
   * @param {Array<Object>} results - 结果数组
   * @returns {Object} 包含 allTopics 和 allAudience 的对象
   */
  collectAvailableTags(results) {
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
  }
}

