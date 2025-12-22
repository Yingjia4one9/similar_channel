// Background service worker
// 处理扩展图标的点击事件（如果需要的话）

chrome.action.onClicked.addListener((tab) => {
  // 如果是在 YouTube 页面，发送消息给 content script
  if (tab.url && tab.url.includes('youtube.com')) {
    chrome.tabs.sendMessage(tab.id, { action: 'toggle-sidebar' });
  } else {
    // 如果不是 YouTube 页面，打开 YouTube
    chrome.tabs.create({ url: 'https://www.youtube.com' });
  }
});

