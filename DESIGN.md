# Axiom Cloud UI Design Spec (Target)

This document defines the **new visual system and page behavior** we will follow going forward.

---

## 1) Design Direction

**Theme:** Axiom Obsidian (dark, technical, premium)

**Goals:**
- High information density without visual clutter
- "Control-room" feel for ML workflows
- Fast scanability using compact typography + strong contrast hierarchy
- Consistent interaction language across datasets, training, leaderboard, deployments

**Mood keywords:** operational, precise, cinematic, enterprise, low-noise

---

## 2) Visual Foundations

### 2.1 Color System

Use dark surfaces with violet primary accents.

- Background: `#0e0e10`
- Base surface: `#19191d`
- Surface variant: `#25252b`
- Primary: `#d0bcff`
- Primary accent/deep: `#6e3bd7`
- Text primary: `#e7e4ec`
- Text muted: `#acaab1`
- Border/outline: `#47474d`

Usage rules:
- Primary color is for active states, key metrics, and CTA emphasis only.
- Avoid bright gradients except subtle tonal overlays.
- Keep most surfaces matte, with occasional glass treatment for focus cards.

### 2.2 Typography

Font family: **Inter**

- Page title: 28–36px, semibold/bold
- Section title: 12–14px uppercase tracking
- Body: 13–14px
- Dense labels/meta: 9–11px uppercase, wide tracking
- Numeric metrics: monospace-style look preferred where possible

### 2.3 Shape, Borders, Depth

- Radius: 8px default (`rounded-lg`), 12px large (`rounded-xl`)
- Border: thin, low-contrast (`border-outline-variant/10~20`)
- Shadows: deep but soft (`void-shadow` style), avoid hard drop shadows
- Blur/glass: use on primary panels only (not every card)

### 2.4 Icons

- Material Symbols Outlined or Lucide (consistent per page)
- Icon size: 16–20 for nav/actions, 24–32 for hero/feature cards

---

## 3) Layout System

### 3.1 Shell

Persistent app frame:
- **Top bar** (fixed): app brand, global actions, account
- **Left sidebar** (fixed): project context + section nav
- **Main canvas**: content area with page-specific grids

### 3.2 Spacing

- Global page padding: 32–48px
- Section gaps: 24–32px
- Card internal padding: 20–32px
- Dense table cell Y padding: 12–20px

### 3.3 Responsiveness

- Desktop-first, but preserve hierarchy on tablet/mobile
- Sidebar collapses/overlays on smaller screens
- Data tables become horizontal scroll blocks

---

## 4) Component Standards

### 4.1 Navigation

- Active sidebar item uses left border + tinted background + primary text
- Inactive items are muted with hover brighten

### 4.2 Cards

Types:
1. Standard surface card
2. Glass panel (focus/critical module)
3. Metric card (large value + tiny label)

### 4.3 Tables

- Header row with compact uppercase labels
- Alternating hover highlight (not zebra striping)
- Right-align numeric columns
- Status as tiny pill/badge

### 4.4 Buttons

- Primary CTA: violet background, high contrast text
- Secondary: surface-high with border
- Tertiary/icon-only for utility actions

### 4.5 Status Tokens

- Active/Healthy: primary or emerald accent
- Processing/Warming: amber/violet pulse
- Idle/Hibernating: muted gray
- Error: soft red, never neon

### 4.6 Charts + Data Viz

- Thin lines, minimal chartjunk
- Muted grids, highlighted primary series
- Small legends with compact labels

---

## 5) Page Blueprints

## 5.1 Datasets

**Purpose:** ingest + registry overview

Sections:
1. Header (title + subtitle)
2. Upload/Ingestion panel (drag/drop + CTA)
3. Storage utilization card
4. Registry table
5. Context cards (auto-labeling, encryption, pipeline sync)

Behavior:
- Upload area supports CSV/XLSX now; UI can reference future formats only if actually supported.
- Table actions are icon-first; detail appears in row/meta panels.

## 5.2 Model Training

**Purpose:** live training cockpit

Sections:
1. Session header with uptime + stop action
2. Left rail: dataset source + hyperparameters + hardware stats
3. Right main: chart + live logs terminal
4. Bottom metrics strip (epoch, ETA, accuracy, cost)

Behavior:
- Training state should feel “live” (animated pulse indicators, rolling logs)
- Controls are editable but grouped and safe (apply/confirm changes)

## 5.3 Leaderboard

**Purpose:** compare models + validate quality

Sections:
1. Header with filter/export controls
2. Main ranked model table
3. Side insights (average accuracy + ROC panel)
4. Secondary diagnostics (confusion matrix + compute logistics)

Behavior:
- Rank and score readability is top priority
- Most important model should be obvious in <2 seconds

## 5.4 Deployments

**Purpose:** runtime endpoint operations

Sections:
1. Header with scale/new deployment actions
2. Endpoints table (status/latency/uptime)
3. Inference sandbox panel
4. Resource metrics and API control
5. Traffic log stream

Behavior:
- Operational safety + transparency over decorative visuals
- High-risk actions should be visually separated and confirmed

---

## 6) Motion & Interaction

- Microtransitions: 120–200ms
- Hover: subtle opacity/background shifts
- Active press: slight scale-down (`~0.95`)
- Pulsing only for truly live states (streaming/training)
- Avoid flashy entrance animations

---

## 7) Implementation Rules (Next.js + Tailwind)

1. Keep all design tokens in Tailwind theme extensions.
2. Reuse existing layout shell (`top nav + sidebar + content`) instead of page-by-page reinvention.
3. Standardize reusable components for:
  - `MetricCard`
  - `StatusBadge`
  - `DataTable`
  - `PanelHeader`
4. Prefer semantic utility groupings over one-off inline styling.
5. Maintain accessibility:
  - visible focus states
  - sufficient contrast
  - keyboard-operable controls

---

## 8) Content Voice

UI copy style:
- concise
- technical but understandable
- no marketing fluff in operational screens

Examples:
- Good: “Training Loss / Validation Accuracy”
- Good: “Cluster: Stable”
- Avoid: overly narrative or hype text in core workflows

---

## 9) Scope Guardrails

This spec is authoritative for visual direction. If current UI differs:
- new work follows this spec,
- legacy UI is migrated incrementally,
- behavior constraints from backend/API still take precedence over visual mockup details.

---

## 10) Migration Plan (Practical)

1. Update global tokens/theme first
2. Apply shell consistency (nav + sidebar + spacing)
3. Migrate pages in order:
  1) Datasets
  2) Training
  3) Leaderboard
  4) Deployments
4. Consolidate repeated UI into shared components
5. QA for contrast, responsiveness, and state clarity
