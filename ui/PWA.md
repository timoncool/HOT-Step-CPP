# PWA and phone sheets

Production builds register `/sw.js` (`src/main.tsx`). Vite generates a worker
identity from the bundle, manifest and icons (the `hotstep-app-shell` plugin in
`vite.config.ts` fills `public/sw.js`'s build and asset placeholders).
Installation precaches the HTML, built assets, manifest and icons; activation
retires only older `hotstep-shell-` caches. Navigations, manifest and icons
revalidate online; offline, a navigation falls back to the cached shell and the
manifest and icons to their cached copies. Non-GET requests, `/api`, `/audio`,
`/references`, `/mm3`, `/socket.io`, `/ws`, media files, range requests and
cross-origin traffic bypass the worker. Offline mode loads the interface;
generation and library requests still need the server.

Completion notifications come from `useJobProgressWatcher`
(`src/hooks/useJobProgress.ts`): when a queue item succeeds while the tab is
hidden, it shows one through the service worker registration, falling back to
a page `Notification`, with the song id as its data. Permission is asked only
from a user gesture: the first Generate click asks once.

Notification clicks focus an existing window and deliver `open-song` after the
mounted listener acknowledges readiness. A new or still-loading window receives
the song through the `open-song` URL parameter, consumed after app effects mount.
The parameter is then removed without changing the current history entry.

Phone song actions use `BottomSheet`; desktop actions keep their dropdown.
`BarSection` keeps its desktop dropdown. The responsive parameter bar already
provides phone sections, so it does not mount a second phone sheet there.
Sheet backdrop/content use z-index 59/60, above settings at 55 and below child
dialogs at 70. While a visible higher portal owns focus, the sheet suspends its
Tab and Escape handlers and backdrop dismissal. Closing the sheet returns focus
to its opener.

## Production fixture

Install Playwright in an isolated ignored directory if it is unavailable:

```powershell
npm install --prefix node_modules/.cache/pwa-tools --no-save --package-lock=false playwright
node scripts/check-pwa.mjs --playwright node_modules/.cache/pwa-tools/node_modules/playwright
```

Chromium must be installed for that Playwright version. The fixture creates two
production bundles in `ui/node_modules/.cache`, changes only copied metadata and
an icon between them, and serves them on an ephemeral loopback port. It never
starts the app server or engine and never uses the shared dev ports or real data.
It closes Chromium/server and removes its generated files on completion.

Checks cover registration, offline HTML and hashed assets, API/audio/range bypass,
metadata revalidation, update activation and namespaced cleanup. The built app
receives notification intents for existing and cold-start windows. Browser
notification events are synthetic; OS focus/open-window permission boundaries
are replaced for this check. A separate built React fixture checks child-dialog
layering, keyboard ownership and focus return using the real `BottomSheet`.

To reproduce the original metadata regression, add
`--worker-ref 1f1c45ae9ab62ea7817ca9ef5c8a01fd259312c3`. This must fail because
the old worker stays identical across the two metadata-only releases.
Use `--sheet-ref` with the same SHA to reproduce Escape closing both the child
dialog and its parent sheet. Neither option changes the shared source files.

The batch runtime owner still checks the integrated app on `:3000`. Real-phone
installation, OS notification clicks and appearance need a secure origin and a
human check. The fixture does not claim those checks.
