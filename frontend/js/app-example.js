/**
 * 应用主文件示例
 * 展示如何使用共享模块重构前端代码
 */

// 假设已经加载了 common.js
// <script src="frontend/js/common.js"></script>

(function() {
  'use strict';

  // 初始化
  const API_BASE = "http://127.0.0.1:8000";
  const api = new YTSimilarFrontend.APIClient(API_BASE);
  const filterSort = new YTSimilarFrontend.FilterSortManager();
  const renderer = YTSimilarFrontend.Renderer;

  let originalResults = [];
  let currentResults = [];

  // 获取DOM元素
  const getEl = id => document.getElementById(id);
  const resultsContainer = getEl("results");
  const resultsCount = getEl("results-count");
  const progressBar = getEl("progress-bar");
  const loadingText = getEl("loading-text");

  // 搜索按钮事件
  function setupSearchButton() {
    getEl("search-btn").addEventListener("click", async () => {
      const url = getEl("channel-url").value.trim();
      if (!url) {
        alert("请输入频道链接");
        return;
      }

      const payload = {
        channel_url: url,
        max_results: Number(getEl("max-results").value || 30),
        min_subscribers: getEl("min-subs").value ? Number(getEl("min-subs").value) : null,
        max_subscribers: getEl("max-subs").value ? Number(getEl("max-subs").value) : null,
        min_similarity: getEl("min-sim").value ? Number(getEl("min-sim").value) : null,
      };

      // 显示加载状态
      resultsContainer.innerHTML = `
        <div class="loading-content">
          <div class="progress-container">
            <div class="progress-bar animated" id="progress-bar"></div>
          </div>
          <div class="loading-text" id="loading-text">正在分析相似频道，请稍候...</div>
        </div>
      `;

      const progressBarEl = getEl("progress-bar");
      const loadingTextEl = getEl("loading-text");

      // 使用共享的 API 客户端
      api.searchSimilarChannelsStream(payload, {
        onProgress: (progress, message) => {
          if (progressBarEl) progressBarEl.style.width = `${progress}%`;
          if (progressBarEl) progressBarEl.classList.remove("animated");
          if (loadingTextEl) loadingTextEl.textContent = message;
        },
        onResult: (data) => {
          originalResults = data.similar_channels || [];
          updateResults();
        },
        onError: (error) => {
          alert(`错误: ${error}`);
          resultsContainer.innerHTML = "";
        }
      });
    });
  }

  // 导出按钮事件
  function setupExportButton() {
    getEl("export-btn").addEventListener("click", async () => {
      const url = getEl("channel-url").value.trim();
      if (!url) {
        alert("请先输入频道链接");
        return;
      }

      const payload = {
        channel_url: url,
        max_results: Number(getEl("max-results").value || 30),
        min_subscribers: getEl("min-subs").value ? Number(getEl("min-subs").value) : null,
        max_subscribers: getEl("max-subs").value ? Number(getEl("max-subs").value) : null,
        min_similarity: getEl("min-sim").value ? Number(getEl("min-sim").value) : null,
      };

      try {
        await api.downloadCSV(payload);
      } catch (err) {
        alert(`导出失败: ${err.message}`);
      }
    });
  }

  // 更新结果显示
  function updateResults() {
    currentResults = filterSort.applyFiltersAndSort(originalResults);
    renderer.renderResults(
      currentResults,
      resultsContainer,
      resultsCount,
      originalResults.length
    );
  }

  // 初始化筛选和排序UI
  function setupFilterSortUI() {
    // 排序按钮
    getEl("sort-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      const menu = getEl("sort-menu");
      menu.classList.toggle("active");
      setTimeout(() => {
        document.addEventListener("click", function closeMenu() {
          menu.classList.remove("active");
          document.removeEventListener("click", closeMenu);
        });
      }, 0);
    });

    // 排序选项
    document.querySelectorAll("#sort-menu .dropdown-item").forEach(item => {
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        const field = item.dataset.sort;
        const currentDir = item.dataset.dir;
        
        if (filterSort.currentSort.field === field) {
          const newDir = currentDir === "asc" ? "desc" : "asc";
          filterSort.setSort(field, newDir);
        } else {
          filterSort.setSort(field, currentDir);
        }
        
        updateResults();
        getEl("sort-menu").classList.remove("active");
      });
    });

    // 筛选按钮
    getEl("filter-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      const menu = getEl("filter-menu");
      menu.classList.toggle("active");
      
      if (menu.classList.contains("active") && originalResults.length > 0) {
        initFilterTags();
      }
      
      setTimeout(() => {
        document.addEventListener("click", function closeMenu() {
          menu.classList.remove("active");
          document.removeEventListener("click", closeMenu);
        });
      }, 0);
    });

    // 应用筛选
    getEl("apply-filter").addEventListener("click", () => {
      filterSort.setFilters({
        minSubs: getEl("filter-min-subs").value ? Number(getEl("filter-min-subs").value) : null,
        maxSubs: getEl("filter-max-subs").value ? Number(getEl("filter-max-subs").value) : null,
      });
      updateResults();
      getEl("filter-menu").classList.remove("active");
    });

    // 重置筛选
    getEl("reset-filter").addEventListener("click", () => {
      filterSort.resetFilters();
      document.querySelectorAll(".filter-tag").forEach(tag => tag.classList.remove("selected"));
      getEl("filter-min-subs").value = "";
      getEl("filter-max-subs").value = "";
      updateResults();
      getEl("filter-menu").classList.remove("active");
    });
  }

  // 初始化筛选标签
  function initFilterTags() {
    if (originalResults.length === 0) return;
    
    const tags = filterSort.collectAvailableTags(originalResults);
    
    // 渲染 Topics 标签
    const topicsContainer = getEl("filter-topics");
    topicsContainer.innerHTML = tags.allTopics.map(topic => 
      `<span class="filter-tag" data-topic="${topic}">${topic}</span>`
    ).join("");
    
    // 渲染 Audience 标签
    const audienceContainer = getEl("filter-audience");
    audienceContainer.innerHTML = tags.allAudience.map(aud => 
      `<span class="filter-tag" data-audience="${aud}">${aud}</span>`
    ).join("");
    
    // 绑定点击事件
    topicsContainer.querySelectorAll(".filter-tag").forEach(tag => {
      tag.addEventListener("click", () => {
        tag.classList.toggle("selected");
        const selected = Array.from(topicsContainer.querySelectorAll(".filter-tag.selected"))
          .map(t => t.dataset.topic);
        filterSort.setFilters({ topics: selected });
        updateResults();
      });
    });
    
    audienceContainer.querySelectorAll(".filter-tag").forEach(tag => {
      tag.addEventListener("click", () => {
        tag.classList.toggle("selected");
        const selected = Array.from(audienceContainer.querySelectorAll(".filter-tag.selected"))
          .map(t => t.dataset.audience);
        filterSort.setFilters({ audience: selected });
        updateResults();
      });
    });
  }

  // 初始化
  function init() {
    setupSearchButton();
    setupExportButton();
    setupFilterSortUI();
  }

  // DOM 加载完成后初始化
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();

