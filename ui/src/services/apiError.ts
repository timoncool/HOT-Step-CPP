// apiError.ts — the one error shape every /api/* failure becomes.
//
// A superset of workflowApi's WorkflowRequestError (status + optional
// currentRevision + reason), so a revisioned caller can switch to this client
// without losing that detail — `instanceof ApiError` is enough, callers don't
// need a second error class. Both transports (axios for JSON, XHR for uploads)
// normalize through it; see axiosClient.ts's handleFailure().

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  /** Set on a 409 caused by a stale revision (revisioned document/section writes). */
  /** Numbered revisions (documents, workflows) or string revisions (training ops). */
  readonly currentRevision?: number | string;
  /** e.g. 'unsupported-version' on a 409 for a stored document this build can't read. */
  readonly reason?: string;
  constructor(status: number, message: string, body: unknown, currentRevision?: number, reason?: string) {
    super(message);
    this.status = status;
    this.body = body;
    this.currentRevision = currentRevision;
    this.reason = reason;
  }
}
