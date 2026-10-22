// Replaced by the production build. Each release precaches its own shell.
const CACHE_PREFIX = 'hotstep-shell-';
const CACHE = CACHE_PREFIX + '__HOTSTEP_BUILD__';
const SHELL = /* __HOTSTEP_ASSETS__ */ [];
const BYPASS = /^\/(api|audio|references|mm3|socket\.io|ws)(\/|$)/;
const MEDIA_EXT = /\.(wav|mp3|flac|ogg|opus|m4a|aac|aiff?|webm|mp4)$/i;

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith(CACHE_PREFIX) && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (BYPASS.test(url.pathname) || MEDIA_EXT.test(url.pathname)) return;
  if (req.headers.has('range') || req.destination === 'audio' || req.destination === 'video') return;

  const navigation = req.mode === 'navigate';
  if (!navigation && !SHELL.includes(url.pathname)) return;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    // Only content-hashed bundle URLs are immutable. Revalidate metadata and icons.
    if (!navigation && url.pathname.startsWith('/assets/')) {
      const hit = await cache.match(req);
      if (hit) return hit;
    }
    try {
      const res = await fetch(req, { cache: navigation || !url.pathname.startsWith('/assets/') ? 'no-cache' : 'default' });
      if (res.ok) await cache.put(navigation ? '/index.html' : req, res.clone());
      return res;
    } catch (err) {
      const hit = await cache.match(navigation ? '/index.html' : req);
      if (hit) return hit;
      throw err;
    }
  })());
});

// A focused client can still be booting. Ask its mounted listener to acknowledge
// readiness before sending the song. Otherwise reload with a startup intent.
async function deliverSong(client, songId) {
  const ready = await new Promise(resolve => {
    const channel = new MessageChannel();
    const finish = value => { clearTimeout(timer); channel.port1.close(); resolve(value); };
    const timer = setTimeout(() => finish(false), 1500);
    channel.port1.onmessage = e => finish(e.data === 'ready');
    client.postMessage({ type: 'notification-ready' }, [channel.port2]);
  });
  if (ready) client.postMessage({ type: 'open-song', songId });
  else {
    const url = new URL(client.url);
    url.searchParams.set('open-song', songId);
    await client.navigate(url.href);
  }
}

self.addEventListener('notificationclick', e => {
  e.notification.close();
  const songId = e.notification.data?.songId;
  e.waitUntil((async () => {
    const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    const client = windows.find(c => c.focused) || windows[0];
    if (client) {
      await client.focus();
      if (typeof songId === 'string' && songId) await deliverSong(client, songId);
    } else {
      const url = new URL('/', self.location.origin);
      if (typeof songId === 'string' && songId) url.searchParams.set('open-song', songId);
      await self.clients.openWindow(url.href);
    }
  })());
});
