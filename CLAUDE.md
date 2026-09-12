# LLM Architecture Seminar Presentation

This repository produces a technical PowerPoint presentation about modern
LLM architecture.

The presentation is for Deep Learning engineers whose primary work is
algorithmic development.

The audience is comfortable with:
- neural networks
- matrix operations
- tensor notation
- optimization
- deep learning architectures

Do not spend slide space explaining basic deep learning concepts unless
they are directly required for an LLM-specific architectural distinction.

---

# Primary Objective

The deck should build accurate and reusable mental models of modern LLM
architecture.

The audience should understand:

1. What computation happens inside an LLM.
2. Why architectural components exist.
3. How training and inference differ architecturally.
4. Which architectural modifications address which bottlenecks.
5. How these concepts connect to later multimodal architectures.

Technical correctness is more important than visual decoration.

---


## Output Target & Syntax
- Target File: `output/deck.md` using MARP Markdown format.
- Title Mandate: Every title must state an explicit takeaway (e.g., "Flow Matching Straightens Trajectories via Linear Vector Fields").
- Math Formatting: Use standard LaTeX `$inline$` and `$$display$$` math.
- Split-Screen Pattern: Split slide content into 2 equal columns (Left: Mathematical Equations; Right: Modular PyTorch code/pseudo-code).

## VS Code Interaction Rules
- Edit ONLY the targeted slide block in `output/deck.md`.
- Never rewrite the entire `deck.md` file unless explicitly instructed.

## Layout & Overflow Constraints
- Code Block Limit: Maximum 8-10 lines of code per code snippet.
- Line Length Limit: Maximum 45 characters per line inside code blocks (force wrapping or shorter variable names).
- Auto-Scaling Code: Always wrap code columns in `<div class="code-col">` to invoke auto-scaling CSS.
- Column Ratio: Use `grid-template-columns: 1fr 1fr;` (50/50 split) with `gap: 1.5rem` to prevent right-column bleed.
- Math Formatting: Keep inline equations compact. Break display math into multiple lines using `\begin{aligned} ... \end{aligned}` if an equation exceeds ~50 characters.

## Theme & CSS Integration Rules
- Stylesheet Path: `@import 'design/theme.css';`
- Global Deck Frontmatter: Every new presentation file in `output/` MUST start with:
  ```yaml
  ---
  marp: true
  paginate: true
  style: "@import '../design/theme.css';"
  ---



# Source of Truth

Do not infer slide intent from the generated PPTX.

The source of truth is:

1. slides/<slide>/contract.md
2. deck/deck_spec.md
3. design/design_system.md
4. design/notation.md

The generated PPTX is an output artifact.

---

# Context Discipline

When working on one slide:

Read only:
- CLAUDE.md
- that slide's contract
- relevant design rules
- that slide's implementation
- the current render if available

Do not inspect unrelated slides unless explicitly requested.

For local edits, do not read the entire deck.

For section or global reviews, explicitly inspect the required slide range.

---

# Slide Editing Rules

Make the smallest change that solves the requested problem.

Do not:
- redesign unrelated slides
- change technical content unless requested
- change terminology globally from a local task
- modify the design system without explicit approval
- regenerate the full deck for a local slide correction

Preserve established visual conventions.

---

# Technical Diagram Rules

Every arrow must represent a meaningful relationship.

Before adding an arrow, determine whether it represents:
- tensor/data flow
- computation
- control/routing
- conceptual dependency

Do not use visually attractive arrows that imply incorrect computation.

Architecture diagrams should prioritize:

1. Correct computation
2. Clear directionality
3. Tensor/component relationships
4. Visual simplicity

Prefer structured vector graphics or PowerPoint primitives for exact
technical diagrams.

Avoid generative imagery for computational diagrams.

---

# Text Rules

Text must not duplicate the diagram.

Prefer:
- short labels
- mathematical notation
- concise technical statements

Avoid:
- paragraph-heavy slides
- generic explanations
- decorative buzzwords
- definitions the audience already knows

Every slide should have one visually dominant takeaway.

---

# Implementation Rules

Treat the presentation as software.

Source files must be modular.

Each slide should be independently editable.

Reusable visual patterns belong in implementation/components/.

Do not duplicate complex diagram logic when a reusable component is
appropriate.

---

# Rendering and QA

After modifying a slide:

1. Build the affected slide/deck.
2. Render the affected slide.
3. Inspect the rendered result.
4. Check against the slide contract.

Check for:
- technical correctness
- semantic correctness of diagrams
- overflow
- clipping
- overlap
- unreadable text
- incorrect visual hierarchy
- misleading arrows
- unnecessary visual complexity

Do not assume generated PPTX code is visually correct.

---

# Review Discipline

Separate:

1. Analysis
2. Proposed changes
3. Implementation
4. Verification

Unless explicitly asked to immediately edit, first analyze and report
issues before making major changes.

---

# Default Priorities

When tradeoffs exist, prioritize:

1. Technical correctness
2. Conceptual clarity
3. Audience-appropriate detail
4. Visual hierarchy
5. Consistency
6. Aesthetics