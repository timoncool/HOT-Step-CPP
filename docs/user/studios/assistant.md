# Assistant

A chat sidebar backed by an LLM you configure, with a live snapshot of your current settings on every message. Ask it to explain a control, recommend settings for a genre, troubleshoot a bad-sounding render, or write and rewrite lyrics and the style caption, then apply what it suggests with one click.

![Assistant](../../images/studio-assistant.webp)
<!-- screenshot: needed -->

## Where to find it

The "Assistant" toggle near the bottom of the sidebar, which opens a chat panel on the right side of the window, beyond the Activity column. Drag its left edge to resize it (320 to 700 pixels); the width and whether the panel is open are remembered.

It needs at least one working LLM provider configured under Settings > AI Services: Gemini, OpenAI, Anthropic, Ollama, LM Studio, Unsloth Studio, llama.cpp, or a custom OpenAI-compatible endpoint. This is the same provider registry Lyric Studio uses, so a provider set up for one works for both. Cloud providers (Gemini, OpenAI, Anthropic) need an API key in the "API Keys" section. The OpenAI-compatible endpoint needs its base URL, and Unsloth Studio needs a username and password or "Keyless local Studio" turned on. Ollama, LM Studio and llama.cpp work from their default local addresses unless you change them, and LM Studio takes an optional key. The Provider dropdown lists every provider and greys out the ones the server reports as unavailable, marked "(unavailable)". Ollama, LM Studio and llama.cpp always count as available, even when they are not running.

## Workflow

1. Open the Assistant from the sidebar.
2. Pick a provider and model from the two dropdowns at the top. The first available provider is auto-selected the first time; after that your choice is remembered.
3. Type a question or request and press Enter (Shift+Enter for a new line). On the welcome screen, a starter suggestion ("Set me up for lo-fi hip hop", "Best solver for clean vocals?", "Review my current settings", "Fix metallic artifacts") puts its question in the box for you to send.
4. The response streams in. If the model produces a reasoning block, it appears above the answer as a collapsible "Thought process" section.
5. If the response includes setting changes, a "Suggested Changes" card lists each one as a from-to row. Apply a single row or click "Apply All"; applied rows get a checkmark.
6. Applying a change takes effect immediately, either as an engine parameter (visible right away in the global parameter bar) or as a content field (caption, lyrics, and the rest, written to the same storage Custom-Gen reads from).

## Controls

| Control | What it does |
|---|---|
| Provider | Which configured LLM answers. Unavailable providers are shown disabled. |
| Model | Which model of the selected provider to use. Switching provider resets it to that provider's default model. |
| Message input | Ask a question or request a change. Enter sends, Shift+Enter adds a line break. Locked while a response is streaming. |
| Clear chat (trash icon) | Discards the conversation shown in the panel and cancels a response that's still streaming. |
| Close (x) | Closes the panel, which also drops the conversation. |
| Suggested Changes card | Appears under a response that included at least one setting change. Shows the current value, the proposed value, and an Apply button per row, plus Apply All with a count; it reads All Applied once every row is in. |
| Thought process | Collapsible block showing the model's reasoning, when the model provides one. Auto-expanded while still streaming. |

## Tips and limits

The assistant can only change the fields the panel knows how to set: models, adapter, adapter scale, mode and group scales, solver, scheduler, guidance mode and the solver and guidance sub-parameters, inference steps, CFG, shift, seed, batch size, LM settings, all of post-processing (spectral lifter, denoiser, mastering, PP-VAE, StableStep, DCW), latent shift and rescale, custom timesteps, duration/auto-trim, and the content fields caption, lyrics, instrumental, bpm, duration, keyScale, timeSignature and vocalLanguage. A suggested change to any other field is left off the card. It cannot start a generation, open the library, install a model, or navigate the UI for you; it only describes, recommends, and sets values you then act on.

Every message resends the whole conversation so far plus a fresh JSON snapshot of your settings. Nothing is kept on the server between requests, and closing the panel or reloading the page drops the conversation; there's no chat history to come back to.

The settings snapshot's content fields (caption, lyrics, and the rest) are read from the same storage the Custom-Gen panel writes to. If you're in a different studio, the assistant still sees the mode you're in, but suggested content edits apply to Custom-Gen's fields, not that studio's own inputs.

The lyric rules (only recognised section tags; text in parentheses is sung as backing vocals) are instructions in the assistant's knowledge base. Nothing checks a suggested lyrics edit against them: Apply writes the text exactly as the model wrote it, so read it before you render.

If a generation or stem job is using the GPU and a local provider needs it too, a chat request can time out. Raise "LLM call timeout (ms)" in Settings > AI Services.

## Related

- [Settings](settings.md)
- [Custom-Gen](create.md)
- [Lyric Studio](lyric-studio.md)
- [Generation](../generation.md)
- [Quality](../quality.md)
