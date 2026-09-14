# Deck Rules

## Conflict Resolution Protocol

- Before executing a user directive, check it against existing rules.
- If a directive conflicts with an existing rule, stop before execution.
- Surface the conflict clearly and ask for a decision.
- Proceed only after the user chooses one of:
	- Overwrite the rule.
	- One-time workaround/exception.
	- Reconsider and revise the directive.
- Do not apply changes until that decision is provided.

## Layout & Header Consistency

- Content-slide titles must sit at a fixed top baseline. Never vertically center a slide's content (e.g. no `justify-content: center` on any content-slide CSS class), even for sparse slides — centering is what causes titles to land at different heights across the deck.
- Handle sparse content by enlarging it (bigger font-size/line-height, e.g. a `.spacious`-style class) and/or merging with a same-subject neighbor. Never fix sparseness by centering content vertically.
- Keep all regular content-slide titles at one consistent heading level (`###` / h3). Reserve `##` / h2 for section-header divider slides only, and `####` / h4 only for a genuine sub-header nested inside a slide that already has its own h3 title.
- Every content slide must have a title. A headerless slide is a defect, not a style choice.
- Watch the right-side accent bar when sizing content: keep the existing side padding so text/images never crowd or run under it, especially in 2-column layouts.

## Slide Merging Rules

- Merging or splitting slides to fix density/overflow is allowed, but only between slides that share the same subject/chapter. Never merge across chapter or subject boundaries.
- Prefer a 2-column layout (`<div class="columns">`) for merges that pair two parallel or sequential pieces of the same idea (formula + worked example, compare/contrast, before/after). Shrink any images enough to sit side by side without overflow.

## Chapter Structure

- Every chapter listed in the Chapter Roadmap gets its own `section-header` divider slide. No chapter may open directly into a regular content slide.
- If a chapter has an intro/bridging blockquote, it belongs on that chapter's own divider slide, not as a separate, near-empty slide placed right after it.

## Diagrams

- Never depend on a Marp `engine:` plugin or an embedded diagram DSL (e.g. Mermaid fenced code blocks) to render a diagram — verify any `engine:` dependency actually exists and installs before relying on it. Pre-render diagrams as standalone SVGs under `assets/diagrams/` and reference them as images (consistent with CLAUDE.md's diagram rules).
- Before creating or overwriting any file under `assets/` (diagrams or otherwise), check whether it already exists and what references it — `git status`, and `grep -r` across `docs/` (including `docs/archive/`) and `output/` — before writing. Never silently overwrite a pre-existing tracked asset; give it a new filename if the old one is still in use elsewhere.

## Draft Hygiene

- No visible "TODO" / "#TODO" authoring markers may remain in shipped slide text or titles. Strip the marker text without inventing replacement content, and flag genuinely missing content to the user rather than filling it in from general knowledge.

## Rendering for QA

- This repo has no system-wide Marp toolchain. To actually render and inspect slides (required by CLAUDE.md's Rendering and QA section) rather than reasoning from source alone: use a local Node.js binary with `@marp-team/marp-cli` installed into a scratch directory, pointed at a Chromium executable (e.g. a cached Playwright/Puppeteer browser, or the VS Code Marp extension's bundled `@marp-team/marp-cli`) via `--browser chrome --browser-path <path>`. Export to PNG per slide (`--images png`) to visually check density, overflow, and header alignment before reporting a layout change as done.
