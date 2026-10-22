// generation/envelope.ts — immutable request snapshot at enqueue time

import { getActiveBackendId, getBackend } from '../backends/registry.js';
import { GENERATION_ENVELOPE_VERSION } from '../backends/types.js';
import { readBackendOverride } from '../../contracts/generation.js';
import type {
  GenerationEnvelope,
  GenerationOperation,
  ResolvedRequest,
} from '../backends/types.js';

export type GenerationEnvelopeErrorCode =
  | 'invalid_body'
  | 'unknown_backend'
  | 'unsupported_operation';

/** Validation failures are safe to return as a client-side 400. An unknown
 * captured backend is kept distinct so it can never be silently defaulted. */
export class GenerationEnvelopeError extends Error {
  readonly code: GenerationEnvelopeErrorCode;

  constructor(code: GenerationEnvelopeErrorCode, message: string) {
    super(message);
    this.name = 'GenerationEnvelopeError';
    this.code = code;
  }
}

function clone<T>(value: T): T {
  try {
    return structuredClone(value);
  } catch (err) {
    throw new GenerationEnvelopeError(
      'invalid_body',
      `Generation request could not be cloned: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function freezeDeep<T>(value: T, seen = new WeakSet<object>()): T {
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return value;
  const object = value as object;
  if (seen.has(object)) return value;
  seen.add(object);
  for (const key of Reflect.ownKeys(object)) {
    freezeDeep((object as Record<PropertyKey, unknown>)[key], seen);
  }
  return Object.freeze(value);
}

function requireBody(raw: unknown, userId: string, jobId: string, enqueuedAt: number): Record<string, unknown> {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new GenerationEnvelopeError('invalid_body', 'Generation request body must be a JSON object');
  }
  if (!userId || !jobId) {
    throw new GenerationEnvelopeError('invalid_body', 'Generation request requires userId and jobId');
  }
  if (!Number.isFinite(enqueuedAt)) {
    throw new GenerationEnvelopeError('invalid_body', 'Generation request enqueuedAt must be finite');
  }
  const body = raw as Record<string, unknown>;
  if (body.backend !== undefined && typeof body.backend !== 'string') {
    throw new GenerationEnvelopeError('invalid_body', 'Generation request backend must be a string when supplied');
  }
  return body;
}

function resolve(
  raw: Readonly<Record<string, unknown>>,
  backendId: string,
  backend: { resolveRequest: (submission: Readonly<Record<string, unknown>>) => ResolvedRequest; operations: readonly GenerationOperation[] },
): ResolvedRequest {
  let resolved: ResolvedRequest;
  try {
    resolved = backend.resolveRequest(raw);
  } catch (err) {
    if (err instanceof GenerationEnvelopeError) throw err;
    throw new GenerationEnvelopeError(
      'invalid_body',
      `Invalid generation request: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  if (!resolved || typeof resolved.operation !== 'string') {
    throw new GenerationEnvelopeError('invalid_body', 'Backend request resolution returned no operation');
  }
  if (!backend.operations.some((operation: GenerationOperation) => operation === resolved.operation)) {
    throw new GenerationEnvelopeError(
      'unsupported_operation',
      `Backend '${backendId}' does not support generation operation '${resolved.operation}'`,
    );
  }
  return resolved;
}

/** Capture the registered active backend exactly once. `raw.backend` is a
 * log-only mismatch field; it never selects a backend. The raw submission and
 * the resolved request are cloned independently before either is frozen, so
 * persisted settings or other shared objects cannot be frozen accidentally. */
export function buildEnvelope(
  raw: unknown,
  userId: string,
  jobId: string,
  enqueuedAt: number,
): Readonly<GenerationEnvelope> {
  const body = requireBody(raw, userId, jobId, enqueuedAt);
  // Opt-in per-job override (a separate field from `raw.backend`, which stays
  // log-only as described above). Absent means the active backend.
  let override: string | undefined;
  try {
    override = readBackendOverride(body);
  } catch (err) {
    throw new GenerationEnvelopeError('invalid_body', err instanceof Error ? err.message : 'Invalid backendOverride');
  }
  if (override !== undefined && !getBackend(override)) {
    throw new GenerationEnvelopeError('invalid_body', `Unknown backendOverride '${override}'`);
  }
  const backendId = override ?? getActiveBackendId();
  const backend = getBackend(backendId);
  if (!backend) {
    throw new GenerationEnvelopeError(
      'unknown_backend',
      `Active generation backend '${backendId}' is not registered`,
    );
  }

  const submission = freezeDeep(clone(body));
  const resolved = clone(resolve(submission, backendId, backend));
  // An override must not quietly change what was asked for: a backend whose
  // resolver ignores `taskType` (YuE2, SA3) would turn a repaint into a
  // text2music render. Without an override this stays as before.
  if (override !== undefined) {
    const requested = typeof body.taskType === 'string' && body.taskType ? body.taskType : 'text2music';
    if (requested !== resolved.operation) {
      throw new GenerationEnvelopeError(
        'unsupported_operation',
        `Backend '${backendId}' does not support generation operation '${requested}'`,
      );
    }
  }
  const mismatch = typeof body.backend === 'string' && body.backend !== backendId
    ? body.backend : undefined;
  const envelope: GenerationEnvelope = {
    version: GENERATION_ENVELOPE_VERSION,
    jobId,
    userId,
    backendId,
    operation: resolved.operation,
    submission,
    common: resolved.common,
    models: resolved.models,
    options: { [backendId]: resolved.options },
    policy: resolved.policy,
    enqueuedAt,
    ...(mismatch === undefined ? {} : { submittedBackendMismatch: mismatch }),
  };
  return freezeDeep(envelope);
}
