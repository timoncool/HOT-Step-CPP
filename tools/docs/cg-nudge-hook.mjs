#!/usr/bin/env node
// cg-nudge-hook.mjs — Claude Code PreToolUse hook (Grep, Bash). Reminds the agent to try codegraph
// before a text search. Reminder only: never blocks, always exits 0.
import fs from 'node:fs';

let p = {};
try { p = JSON.parse(fs.readFileSync(0, 'utf8') || '{}'); } catch {}
const cmd = p.tool_input?.command;
if (p.tool_name === 'Bash' && !/^\s*(rg|grep|findstr)\b/.test(cmd ?? '')) process.exit(0);

process.stdout.write(JSON.stringify({
  hookSpecificOutput: {
    hookEventName: 'PreToolUse',
    additionalContext: 'Codegraph first (AGENTS.md): for callers, callees, definitions or change impact run `tools\\cg.cmd callers|callees|impact|node|query <symbol>` before grep. Grep is fine for literal text and filenames.',
  },
}));
