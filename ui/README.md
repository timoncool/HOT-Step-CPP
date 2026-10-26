# HOT-Step CPP UI

React 19, Vite and Zustand frontend, with TanStack Query owning server state, served by the Node
server in production and by the Vite dev server on port 3000 in development (`dev.bat`).

- Layout, stores and the studio-folder convention: [docs/dev/architecture.md](../docs/dev/architecture.md)
- Build, dev loop and type checking (`npx tsc --noEmit -p tsconfig.app.json`, not a bare `tsc`): [docs/dev/building.md](../docs/dev/building.md)
- Adding or changing a studio: [.claude/skills/ui-feature-dev/SKILL.md](../.claude/skills/ui-feature-dev/SKILL.md)
- Dropdowns, toggles, labels and the other shared controls: [docs/dev/ui-design.md](../docs/dev/ui-design.md)
- Server state (query keys, the axios transport, cache invalidation): [docs/dev/ui-server-state.md](../docs/dev/ui-server-state.md), with open migration work in [docs/dev/tanstack-query.md](../docs/dev/tanstack-query.md)
- Service worker, notifications and phone sheets: [PWA.md](PWA.md)
- Every studio has a user page under [docs/user/studios/](../docs/user/studios/); update it with the change.
