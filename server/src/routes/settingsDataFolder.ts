import { Router } from 'express';
import path from 'path';
import type { DataLocation, OperationPreview, ScheduleKind } from '../dataroot/pointer.js';
import type { PendingOperation } from '../dataroot/paths.js';

interface Dependencies {
  location(): DataLocation;
  blockers(): string[];
  preview(kind: ScheduleKind, target: string): OperationPreview;
  schedule(kind: ScheduleKind, target: string): PendingOperation;
  cancel(): boolean;
  restart(): void;
}

/** The scheduling write completes before the graceful restart is requested. */
export function createDataFolderHandlers(deps: Dependencies) {
  const handle = (fn: (req: any, res: any) => void) => (req: any, res: any) => {
    try { fn(req, res); } catch (err: any) {
      res.status(err.status || 500).json({ error: err.message, blockers: err.blockers, errors: err.errors });
    }
  };
  function operation(body: any): { kind: ScheduleKind; target: string } {
    if (!body || !['move', 'repoint', 'create', 'import'].includes(body.kind) ||
        typeof body.target !== 'string' || !path.isAbsolute(body.target.trim())) {
      throw Object.assign(new Error('Specify move, repoint, create or import and an absolute folder path'), { status: 400 });
    }
    return { kind: body.kind, target: body.target.trim() };
  }
  const location = handle((_req, res) => res.json({ ...deps.location(), blockers: deps.blockers() }));
  const preview = handle((req, res) => {
    const { kind, target } = operation(req.body);
    res.json({ ...deps.preview(kind, target), blockers: deps.blockers() });
  });
  const schedule = handle((req, res) => {
    const { kind, target } = operation(req.body);
    const pending = deps.schedule(kind, target);
    try { deps.restart(); } catch (err: any) {
      res.status(503).json({ error: `Change saved, but restart failed: ${err.message}. Restart the app to continue.`, pending });
      return;
    }
    res.json({ pending, restartRequested: true });
  });
  const cancel = handle((_req, res) => res.json({ cancelled: deps.cancel() }));
  return { location, preview, schedule, cancel };
}

/** Disposable HTTP fixtures use the same handlers as Settings. */
export function createDataFolderRouter(deps: Dependencies): Router {
  const router = Router();
  const handlers = createDataFolderHandlers(deps);
  router.get('/', handlers.location);
  router.post('/preview', handlers.preview);
  router.post('/schedule', handlers.schedule);
  router.post('/cancel', handlers.cancel);
  return router;
}
