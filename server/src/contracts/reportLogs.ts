// contracts/reportLogs.ts — wire shape for GET /api/logs/snapshot (server/src/routes/logs.ts).

export interface ReportLogsResponse {
  /** Newest-last log lines per source, as the ring buffer holds them (unredacted; the UI redacts). */
  server: string[];
  engine: string[];
  /** This machine's home directory and login name, so the UI can redact them. */
  home: string;
  user: string;
}
