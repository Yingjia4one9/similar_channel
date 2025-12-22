// Content script - 在 YouTube 页面上注入侧边栏

(function() {
  'use strict';

  const API_BASE = "http://127.0.0.1:8000";
  let sidebar = null;
  let sidebarVisible = false;
  let iframe = null;

  // 从 URL 提取频道 ID
  function extractChannelIdFromUrl() {
    const url = window.location.href;
    
    // 匹配 /channel/UC... 格式
    const channelMatch = url.match(/\/channel\/([a-zA-Z0-9_-]+)/);
    if (channelMatch) {
      return channelMatch[1];
    }
    
    // 匹配 /@username 格式
    const handleMatch = url.match(/\/@([a-zA-Z0-9_-]+)/);
    if (handleMatch) {
      // 需要转换为频道 ID，这里先返回 handle
      return handleMatch[1];
    }
    
    // 匹配 /c/... 格式
    const cMatch = url.match(/\/c\/([a-zA-Z0-9_-]+)/);
    if (cMatch) {
      return cMatch[1];
    }
    
    // 尝试从页面元素获取
    const channelLink = document.querySelector('a[href*="/channel/"]');
    if (channelLink) {
      const match = channelLink.href.match(/\/channel\/([a-zA-Z0-9_-]+)/);
      if (match) return match[1];
    }
    
    return null;
  }

  // 创建侧边栏
  function createSidebar() {
    if (sidebar) return;

    // 获取 sidebar.html 的 URL
    const sidebarUrl = chrome.runtime.getURL('sidebar.html');

    // 创建侧边栏容器
    sidebar = document.createElement('div');
    sidebar.id = 'yt-similar-sidebar';
    sidebar.innerHTML = `
      <div class="sidebar-header">
        <div class="sidebar-title">
          <span class="sidebar-icon">🔍</span>
          <span>Similar Channels</span>
        </div>
        <button class="sidebar-toggle" id="sidebar-close-btn">×</button>
      </div>
      <div class="sidebar-content" id="sidebar-content">
        <iframe id="sidebar-iframe" src="${sidebarUrl}" allow="same-origin"></iframe>
      </div>
    `;

    document.body.appendChild(sidebar);
    iframe = document.getElementById('sidebar-iframe');

    // 等待 iframe 加载完成
    iframe.addEventListener('load', () => {
      console.log('Sidebar iframe loaded');
      // 如果侧边栏已打开，自动搜索当前频道
      if (sidebarVisible) {
        setTimeout(() => {
          const channelId = extractChannelIdFromUrl();
          if (channelId) {
            sendMessageToSidebar({
              type: 'auto-search',
              channelId: channelId,
              channelUrl: window.location.href
            });
          }
        }, 500);
      }
    });

    // 绑定关闭按钮
    document.getElementById('sidebar-close-btn').addEventListener('click', toggleSidebar);

    // 调整页面布局
    adjustPageLayout();
  }

  // 调整页面布局以适应侧边栏
  function adjustPageLayout() {
    const mainContent = document.querySelector('#primary, #contents, ytd-watch-flexy');
    if (mainContent) {
      if (sidebarVisible) {
        mainContent.style.marginRight = '400px';
        mainContent.style.transition = 'margin-right 0.3s ease';
      } else {
        mainContent.style.marginRight = '0';
      }
    }
  }

  // 切换侧边栏显示/隐藏
  function toggleSidebar() {
    sidebarVisible = !sidebarVisible;
    
    if (sidebar) {
      sidebar.classList.toggle('visible', sidebarVisible);
    } else {
      createSidebar();
      sidebarVisible = true;
      sidebar.classList.add('visible');
    }
    
    adjustPageLayout();
    
    // 保存状态
    chrome.storage.local.set({ sidebarVisible: sidebarVisible });

    // 如果显示侧边栏，自动检测当前频道
    if (sidebarVisible) {
      setTimeout(() => {
        const channelId = extractChannelIdFromUrl();
        if (channelId) {
          sendMessageToSidebar({
            type: 'auto-search',
            channelId: channelId,
            channelUrl: window.location.href
          });
        }
      }, 100);
    }
  }

  // 向侧边栏发送消息
  function sendMessageToSidebar(message) {
    if (iframe && iframe.contentWindow) {
      try {
        iframe.contentWindow.postMessage(message, '*');
      } catch (e) {
        console.error('Failed to send message to sidebar:', e);
      }
    }
  }

  // 创建工具栏按钮
  function createToolbarButton() {
    // 检查是否已存在按钮
    if (document.getElementById('yt-similar-toggle-btn')) return;

    // 尝试在 YouTube 工具栏添加按钮
    const toolbar = document.querySelector('#masthead-container, #header, ytd-masthead, #container');
    if (!toolbar) {
      // 如果找不到工具栏，延迟重试
      setTimeout(createToolbarButton, 1000);
      return;
    }

    const button = document.createElement('button');
    button.id = 'yt-similar-toggle-btn';
    button.className = 'yt-similar-toggle-btn';
    button.innerHTML = '🔍 Similar';
    button.title = 'Toggle Similar Channels';
    button.style.cssText = `
      background: #9333ea;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      margin-left: 8px;
      transition: background 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
      z-index: 10000;
      position: relative;
    `;
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      toggleSidebar();
    });

    // 尝试插入到合适的位置
    const searchContainer = document.querySelector('#search, #search-form, ytd-searchbox, #search-icon-legacy');
    if (searchContainer && searchContainer.parentElement) {
      searchContainer.parentElement.insertBefore(button, searchContainer.nextSibling);
    } else {
      // 尝试插入到工具栏的末尾
      const endButton = document.querySelector('#end, ytd-topbar-menu-button-renderer');
      if (endButton && endButton.parentElement) {
        endButton.parentElement.insertBefore(button, endButton);
      } else {
        toolbar.appendChild(button);
      }
    }
  }

  // 监听来自侧边栏的消息
  window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'sidebar-ready') {
      iframe = document.getElementById('sidebar-iframe');
      // 如果侧边栏已打开，自动搜索当前频道
      if (sidebarVisible) {
        const channelId = extractChannelIdFromUrl();
        if (channelId) {
          sendMessageToSidebar({
            type: 'auto-search',
            channelId: channelId,
            channelUrl: window.location.href
          });
        }
      }
    }
  });

  // 恢复侧边栏状态
  chrome.storage.local.get(['sidebarVisible'], (result) => {
    if (result.sidebarVisible) {
      createSidebar();
      sidebarVisible = true;
      setTimeout(() => {
        if (sidebar) sidebar.classList.add('visible');
        adjustPageLayout();
      }, 100);
    }
  });

  // 监听来自 background script 的消息
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'toggle-sidebar') {
      toggleSidebar();
      sendResponse({ success: true });
    }
    return true;
  });

  // 初始化
  function init() {
    createToolbarButton();
    
    // 监听 URL 变化（YouTube 使用 SPA）
    let lastUrl = location.href;
    const urlObserver = new MutationObserver(() => {
      const url = location.href;
      if (url !== lastUrl) {
        lastUrl = url;
        // URL 变化时，如果侧边栏打开，自动搜索新频道
        if (sidebarVisible && sidebar) {
          setTimeout(() => {
            const channelId = extractChannelIdFromUrl();
            if (channelId) {
              sendMessageToSidebar({
                type: 'auto-search',
                channelId: channelId,
                channelUrl: url
              });
            }
          }, 500);
        }
      }
    });
    
    urlObserver.observe(document, { subtree: true, childList: true });
    
    // 也监听 popstate 事件（浏览器前进/后退）
    window.addEventListener('popstate', () => {
      if (sidebarVisible && sidebar) {
        setTimeout(() => {
          const channelId = extractChannelIdFromUrl();
          if (channelId) {
            sendMessageToSidebar({
              type: 'auto-search',
              channelId: channelId,
              channelUrl: window.location.href
            });
          }
        }, 500);
      }
    });
  }

  // 等待页面加载完成
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

