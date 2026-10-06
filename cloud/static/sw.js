// TRADECORE service worker: shows the trade alerts the server pushes, even when the app is closed.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});     // lets browsers offer "Install app"

self.addEventListener('push', e => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (_) { d = {body: e.data ? e.data.text() : ''}; }
  e.waitUntil(self.registration.showNotification(d.title || 'TRADECORE', {
    body: d.body || '', tag: (d.tag || 'tradecore') + '-' + Date.now(), icon: '/icon-192.png', badge: '/icon-192.png',
    requireInteraction: true, vibrate: [200, 100, 200, 100, 300], data: {url: d.url || '/'}
  }));
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then(ws => {
    for (const w of ws) { if ('focus' in w) return w.focus(); }
    return self.clients.openWindow((e.notification.data && e.notification.data.url) || '/');
  }));
});
