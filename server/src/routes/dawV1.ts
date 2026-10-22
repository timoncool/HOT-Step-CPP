// routes/dawV1.ts - /api/v1/daw: a small, versioned facade for DAW clients.
//
// Every response carries api:1; errors use { api, error: { code, message } }.
// With HOTSTEP_DAW_KEY set, every caller (loopback and proxied included) must send a
// matching X-DAW-Key; with none set the facade answers only demonstrably local callers.

import { Router, type Request, type Response, type NextFunction } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import {
  cancelJob, checkUploadMagic, dawAuthError, dawKey, getDitInfo, getJob, getStatus, getUnderstand, newId, newTmpDir,
  startSweeper, startUnderstand, submitGenerate, takeWavPath,
  jobEtaS,
} from '../services/daw/dawFacade.js';
import { DawError, errorBody, modeNeedsSource, parseDawRequest, resolveDitModel, toGenerateParams } from '../services/daw/dawSchema.js';

const router = Router();
startSweeper();

router.use((req: Request, res: Response, next: NextFunction) => {
  const err = dawAuthError(dawKey(), req.get('X-DAW-Key'), req.socket.remoteAddress,
    { xff: req.get('x-forwarded-for'), realIp: req.get('x-real-ip'), forwarded: req.get('forwarded') });
  if (err) { res.status(401).json(errorBody('unauthorized', err)); return; }
  next();
});

/** Mounted by index.ts after the app-wide body parsers: a malformed or oversized body gets the DAW envelope. */
export function dawBodyErrorHandler(err: any, _req: Request, res: Response, next: NextFunction): void {
  if (res.headersSent) { next(err); return; }
  const status = Number.isInteger(err?.status) && err.status >= 400 && err.status < 500 ? err.status : 400;
  const message = err?.type === 'entity.parse.failed' ? 'request body is not valid JSON' : String(err?.message ?? 'bad request body');
  res.status(status).json(errorBody('bad_request', message));
}

const AUDIO_EXTS = new Set(['.wav', '.flac', '.aiff', '.aif']);

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, _file, cb) => cb(null, newTmpDir((req as any).dawId)),
    filename: (_req, file, cb) => cb(null, `${file.fieldname}${path.extname(file.originalname).toLowerCase()}`),
  }),
  limits: { fileSize: 64 * 1024 * 1024, files: 2 },
  fileFilter: (_req, file, cb) => {
    if (AUDIO_EXTS.has(path.extname(file.originalname).toLowerCase())) cb(null, true);
    else cb(new DawError(400, 'bad_request', `unsupported audio type ${file.originalname} (wav, flac, aiff)`));
  },
});

const assignId = (req: Request, _res: Response, next: NextFunction) => { (req as any).dawId = newId(); next(); };

function fail(res: Response, err: any): void {
  if (err instanceof DawError) { res.status(err.status).json(errorBody(err.code, err.message)); return; }
  if (err instanceof multer.MulterError) { res.status(400).json(errorBody('bad_request', err.message)); return; }
  res.status(502).json(errorBody('upstream_error', err?.message ?? String(err)));
}

const wrap = (fn: (req: Request, res: Response) => Promise<void>) =>
  (req: Request, res: Response) => { fn(req, res).catch((e) => fail(res, e)); };

/** Run multer and route its errors through the envelope. */
const withUpload = (fields: multer.Field[]) => (req: Request, res: Response, next: NextFunction) =>
  upload.fields(fields)(req, res, (err?: unknown) => {
    if (err) { fail(res, err); return; }
    try {
      for (const list of Object.values((req.files as Record<string, Express.Multer.File[]> | undefined) ?? {})) {
        for (const f of list) checkUploadMagic(f.path);
      }
    } catch (e) { fail(res, e); return; }
    next();
  });

const fileOf = (req: Request, field: string): Express.Multer.File | undefined =>
  (req.files as Record<string, Express.Multer.File[]> | undefined)?.[field]?.[0];

router.get('/status', wrap(async (_req, res) => { res.json(await getStatus()); }));

router.post('/understand', assignId, withUpload([{ name: 'audio', maxCount: 1 }]), wrap(async (req, res) => {
  const f = fileOf(req, 'audio');
  if (!f) throw new DawError(400, 'bad_request', 'multipart field "audio" is required');
  const { id, result } = await startUnderstand((req as any).dawId, f.path);
  if (result) res.json(result);
  else res.status(202).json({ api: 1, analysis_id: id, state: 'running' });
}));

router.get('/understand/:id', wrap(async (req, res) => { res.json(getUnderstand(String(req.params.id))); }));

router.post('/generate', assignId, withUpload([{ name: 'src_audio', maxCount: 1 }, { name: 'ref_audio', maxCount: 1 }]), wrap(async (req, res) => {
  let raw: unknown = req.body?.request;
  if (typeof raw === 'string') {
    try { raw = JSON.parse(raw); } catch { throw new DawError(400, 'bad_request', 'request is not valid JSON'); }
  } else if (raw === undefined) raw = req.body;
  const dawReq = parseDawRequest(raw);
  const src = fileOf(req, 'src_audio');
  if (modeNeedsSource(dawReq.mode) && !src) throw new DawError(400, 'bad_request', `mode ${dawReq.mode} needs src_audio`);
  if (dawReq.mode === 'lego' || dawReq.model !== undefined) {
    const { dit, loaded } = await getDitInfo();
    dawReq.dit_model = resolveDitModel(dawReq, dit, loaded);
  }
  const params = toGenerateParams(dawReq, src?.path);
  const ref = fileOf(req, 'ref_audio');
  if (ref) params.timbreReference = ref.path;
  const id = await submitGenerate((req as any).dawId, dawReq, params);
  res.status(202).json({ api: 1, id, state: 'queued', eta_s: jobEtaS(id), dit_model: dawReq.dit_model ?? null });
}));

router.get('/jobs/:id', wrap(async (req, res) => { res.json(await getJob(String(req.params.id))); }));

router.get('/jobs/:id/audio', wrap(async (req, res) => {
  const n = req.query.take === undefined ? 0 : Number(req.query.take);
  if (!Number.isInteger(n) || n < 0) throw new DawError(400, 'bad_request', 'take must be a non-negative integer');
  const file = await takeWavPath(String(req.params.id), n);
  const size = fs.statSync(file).size;
  res.setHeader('Content-Type', 'audio/wav');
  res.setHeader('Content-Length', String(size));
  res.setHeader('Content-Disposition', `attachment; filename="take_${n}.wav"`);
  fs.createReadStream(file).pipe(res);
}));

router.post('/jobs/:id/cancel', wrap(async (req, res) => { res.json(await cancelJob(String(req.params.id))); }));

router.use((_req: Request, res: Response) => { res.status(404).json(errorBody('not_found', 'unknown DAW endpoint')); });

export default router;
