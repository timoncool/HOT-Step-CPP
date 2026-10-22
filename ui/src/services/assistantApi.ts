import { apiClient, type ReadOptions } from './httpClient';
// assistantApi.ts — Frontend API client for the AI Assistant
//
// Provides streaming chat via SSE (fetch + ReadableStream) and
// provider listing. Same pattern as Lyric Studio's streaming calls.

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantAction {
  set: string;
  value: any;
}

export interface ChatStreamParams {
  message: string;
  history: ChatMessage[];
  currentSettings: Record<string, any>;
  provider: string;
  model?: string;
}

export interface AssistantProvider {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

/**
 * Stream a chat response from the assistant.
 * Returns an AbortController so the caller can cancel mid-stream.
 */
export function chatStream(
  params: ChatStreamParams,
  onChunk: (text: string) => void,
  onComplete: (fullText: string) => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        onError(err.error || `HTTP ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        onError('No response body');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            // Store event type for the next data line
            (reader as any).__lastEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const eventType = (reader as any).__lastEvent || 'chunk';
            const dataStr = line.slice(6).trim();
            try {
              const data = JSON.parse(dataStr);
              if (eventType === 'chunk') {
                onChunk(data.text || '');
              } else if (eventType === 'complete') {
                onComplete(data.text || '');
              } else if (eventType === 'error') {
                onError(data.error || 'Unknown error');
              }
            } catch {
              // Ignore malformed JSON
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Connection failed');
      }
    }
  })();

  return controller;
}

/**
 * Parse action blocks from assistant response text.
 * Actions are embedded in fenced code blocks with the "actions" language tag.
 */
export function parseActions(text: string): AssistantAction[] {
  const regex = /```actions\s*\n([\s\S]*?)```/g;
  const actions: AssistantAction[] = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1].trim());
      if (Array.isArray(parsed)) {
        actions.push(...parsed.filter(a => a && typeof a.set === 'string'));
      }
    } catch {
      // Ignore malformed action blocks
    }
  }
  return actions;
}

/**
 * Strip action blocks from text to get the display-only content.
 */
export function stripActionBlocks(text: string): string {
  return text.replace(/```actions\s*\n[\s\S]*?```/g, '').trim();
}

/**
 * Strip LLM thinking/reasoning blocks from text.
 * Matches the same patterns as the server-side stripThinkingBlocks in postprocess.ts.
 * Handles: <think>, <analysis>, <reasoning>, <reflection>, <thought>, <|channel>thought
 */
export function stripThinkingBlocks(text: string): string {
  // Closed thinking tags (complete blocks)
  let result = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  result = result.replace(/<analysis>[\s\S]*?<\/analysis>/g, '');
  result = result.replace(/<reasoning>[\s\S]*?<\/reasoning>/g, '');
  result = result.replace(/<reflection>[\s\S]*?<\/reflection>/g, '');
  result = result.replace(/<thought>[\s\S]*?<\/thought>/g, '');
  result = result.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '');

  // Unclosed thinking tags (stream still going — strip from open tag to end)
  result = result.replace(/<(?:think|analysis|reasoning|reflection|thought)>[\s\S]*/g, '');
  result = result.replace(/<\|channel>thought[\s\S]*/g, '');

  // Strip CoT-style headers (e.g. "Thinking Process:\n...\n---")
  const cotMatch = result.match(/^(?:\s*\*+\s*)?(?:Thinking Process|Thought Process|Thinking|Reasoning):\s*[\s\S]*?(?:---|[*]{3,}|={3,})\s*/i);
  if (cotMatch) result = result.slice(cotMatch[0].length);

  return result.trim();
}

/**
 * Extract thinking content and response separately from LLM output.
 * Returns both parts so the UI can render thinking in a distinct visual style.
 */
export function extractThinkingAndResponse(text: string): { thinking: string | null; response: string } {
  // Try each known thinking tag format
  const patterns: { open: RegExp; close: RegExp }[] = [
    { open: /<think>/,           close: /<\/think>/ },
    { open: /<thought>/,         close: /<\/thought>/ },
    { open: /<analysis>/,        close: /<\/analysis>/ },
    { open: /<reasoning>/,       close: /<\/reasoning>/ },
    { open: /<reflection>/,      close: /<\/reflection>/ },
    { open: /<\|channel>thought/, close: /<channel\|>/ },
  ];

  for (const { open, close } of patterns) {
    const openMatch = open.exec(text);
    if (!openMatch) continue;

    const afterOpen = text.slice(openMatch.index + openMatch[0].length);
    const closeMatch = close.exec(afterOpen);

    if (closeMatch) {
      // Complete thinking block — extract content between tags
      const thinking = afterOpen.slice(0, closeMatch.index).trim();
      const beforeThink = text.slice(0, openMatch.index).trim();
      const afterThink = afterOpen.slice(closeMatch.index + closeMatch[0].length).trim();
      const response = (beforeThink + ' ' + afterThink).trim();
      return { thinking: thinking || null, response };
    } else {
      // Unclosed tag — still streaming thinking, everything after tag is thinking
      const thinking = afterOpen.trim();
      const response = text.slice(0, openMatch.index).trim();
      return { thinking: thinking || null, response };
    }
  }

  return { thinking: null, response: text };
}

/**
 * Fetch available LLM providers for the assistant.
 */
export async function getProviders(opts?: ReadOptions): Promise<AssistantProvider[]> {
  return apiClient.get<AssistantProvider[]>('/assistant/providers', opts);
}
