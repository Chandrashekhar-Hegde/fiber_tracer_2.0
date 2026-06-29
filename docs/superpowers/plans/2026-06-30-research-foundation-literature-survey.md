# Research Foundation Literature Survey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a publication-grade `docs/RESEARCH_FOUNDATION.md` literature survey (13 sections, ~50–80 verified references) and propagate its references across the existing docs and roadmap epics.

**Architecture:** Theme-scoped research subagents gather and verify literature in waves; the maintainer synthesizes their drafts into one document, assembles a deduplicated reference list, then propagates curated references into the existing docs and posts research-basis comments to epics #18–#21.

**Tech Stack:** Markdown docs; `WebSearch`/`WebFetch` (deferred — load via ToolSearch) for discovery and verification; `Agent` research subagents; `gh` for issue comments.

**Source spec:** `docs/superpowers/specs/2026-06-30-research-foundation-literature-survey-design.md`

## Global Constraints

- No Claude/AI attribution anywhere — doc text, references, or commit trailers (project memory `no-claude-attribution`).
- Canonical references: **50–80 total**, focused depth, emphasis **2020–2026 plus seminal works**.
- **Citation-integrity:** every canonical reference must be confirmed by fetching a real page (publisher / arXiv / Crossref / DOI resolver) matching authors/title/year/venue/DOI. Unverified references are excluded from the canonical list (optional "to verify" appendix only).
- Standards cited by official designation only (e.g., ASTM E1441, ISO 15708); do not invent acceptance thresholds.
- Neutral scholarly tone; no marketing language.
- All work on branch `docs/research-foundation-survey` (created from the spec branch); commit frequently.

## File Structure

- **Create:** `docs/RESEARCH_FOUNDATION.md` — the survey (13 sections).
- **Working registry (not committed):** `<scratchpad>/refs.md` — dedup workspace for references during execution; canonical references live in §13.
- **Modify:** `docs/CITATIONS.md`, `docs/methodology.md`, `docs/MODEL_CARD.md`, `docs/CAPABILITIES.md`, `README.md`, `ROADMAP.md`.

---

### Task 1: Scaffold the survey document

**Files:**
- Create: `docs/RESEARCH_FOUNDATION.md`

**Interfaces:**
- Produces: the section skeleton (anchors §1–§13) that later tasks fill in.

- [ ] **Step 1: Create the branch**

```bash
git checkout docs/research-foundation-spec
git checkout -b docs/research-foundation-survey
```

- [ ] **Step 2: Write the skeleton**

Create `docs/RESEARCH_FOUNDATION.md` with a title, a one-paragraph scope note, and the 13 section headings from the spec (`## 1. Scope & motivation` … `## 13. References`), each with an HTML comment placeholder `<!-- drafted in Task N -->`.

- [ ] **Step 3: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Scaffold research foundation survey document"
```

---

### Task 2: Research wave 1 — standards, acquisition/classical segmentation, DL segmentation

**Files:**
- Modify: `docs/RESEARCH_FOUNDATION.md` (§3, §4, §5)

**Interfaces:**
- Produces: drafted §3, §4, §5 and a growing reference registry (`<scratchpad>/refs.md`).

- [ ] **Step 1: Load web tools**

`ToolSearch` query: `select:WebSearch,WebFetch`

- [ ] **Step 2: Dispatch three research subagents in parallel**

Use the Agent tool (Explore or general-purpose) — one call each, same message. Each prompt ends with this shared contract:

> Return: (1) a 300–600 word neutral scholarly draft of the section (no first person, no AI/author attribution); (2) a reference list where each entry has authors, year, title, venue, DOI, source URL, and a `verified: yes/no` flag — set `yes` ONLY for entries whose metadata you confirmed by fetching the real page (publisher/arXiv/Crossref/DOI). Do not include any reference you could not fetch. (3) 2–4 open research gaps. Emphasize 2020–2026 plus seminal works.

Agent A1 — *Standards & qualification*: "Survey aerospace/defense standards and qualification practice for X-ray CT inspection of fiber-reinforced composites: CMH-17, ASTM E1441/E1570/E1695, ISO 15708, NADCAP, and void/porosity acceptance. …<shared contract>"

Agent A2 — *Acquisition & classical thresholding/segmentation*: "Survey XCT acquisition artifacts (beam hardening, ring, partial-volume) and classical thresholding/segmentation for composite CT (Otsu, adaptive/Sauvola, multi-Otsu, watershed). …<shared contract>"

Agent A3 — *DL & foundation-model segmentation*: "Survey deep-learning segmentation for fiber/composite CT (U-Net, nnU-Net, 3D CNNs) and foundation/promptable models (SAM-family, self-supervised) relevant to fiber segmentation. …<shared contract>"

- [ ] **Step 3: Spot-verify references**

For each returned section, pick 2 references and confirm with `WebFetch` that the DOI/page matches. Drop any that fail.

- [ ] **Step 4: Write sections and registry**

Insert the cleaned drafts into §3, §4, §5 of `docs/RESEARCH_FOUNDATION.md`. Append all `verified: yes` references to `<scratchpad>/refs.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Draft survey sections: standards, acquisition, segmentation"
```

---

### Task 3: Research wave 2 — orientation/morphometry/tracking, DVC/DIC, prediction

**Files:**
- Modify: `docs/RESEARCH_FOUNDATION.md` (§6, §7, §8, §9)

**Interfaces:**
- Consumes: registry from Task 2.
- Produces: drafted §6–§9; expanded registry.

- [ ] **Step 1: Dispatch three research subagents in parallel** (same shared contract as Task 2)

Agent A4 — *Orientation, morphometry, fibre tracking*: "Survey fiber orientation estimation (gradient structure tensor, Advani–Tucker second-order tensor, PCA), morphometry (diameter, length, tortuosity), and skeleton/centerline network analysis in composite CT. …<shared contract>"

Agent A5 — *DVC & DIC*: "Survey Digital Volume Correlation (3D) and Digital Image Correlation (2D) for displacement/strain measurement in composites: methods (local/global, FFT, optical-flow), open libraries (e.g. spam, muDIC), and validation. …<shared contract>"

Agent A6 — *Prediction*: "Survey prediction of composite mechanical behavior from microstructure: micromechanics/homogenization, ML surrogate models, FE coupling, failure/fatigue prediction, and uncertainty quantification. …<shared contract>"

- [ ] **Step 2: Spot-verify references** (2 per section via `WebFetch`; drop failures)

- [ ] **Step 3: Write sections and registry** (§6–§9; append verified refs)

- [ ] **Step 4: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Draft survey sections: orientation/tracking, DVC/DIC, prediction"
```

---

### Task 4: Research wave 3 — digital twins & cross-cutting

**Files:**
- Modify: `docs/RESEARCH_FOUNDATION.md` (§10, §11)

- [ ] **Step 1: Dispatch one research subagent** (shared contract)

Agent A7 — *Digital twins + cross-cutting*: "Survey digital twins for composite materials/structures (definition, data assimilation, model updating, as-manufactured-to-as-designed) and cross-cutting concerns: domain adaptation/generalization across scanners/materials, validation & benchmarking, reproducibility & traceability. …<shared contract>"

- [ ] **Step 2: Spot-verify references** (2 via `WebFetch`)

- [ ] **Step 3: Write §10, §11 and append registry**

- [ ] **Step 4: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Draft survey sections: digital twins and cross-cutting"
```

---

### Task 5: Write framing and synthesis (§1, §2, §12)

**Files:**
- Modify: `docs/RESEARCH_FOUNDATION.md` (§1, §2, §12)

- [ ] **Step 1: Write §1 Scope & motivation and §2 Background**

Author the high-reliability (aerospace/defense) motivation and the composites/XCT background, citing only references already in the registry.

- [ ] **Step 2: Write §12 Synthesis & gap analysis**

Describe the analysis→prediction→digital-twin continuum and a gap table mapping gaps to epics #18 (segmentation quality), #19 (DVC), #20 (DIC), #21 (digital twin).

- [ ] **Step 3: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Add scope, background, and synthesis sections"
```

---

### Task 6: Assemble and verify the canonical reference list (§13)

**Files:**
- Modify: `docs/RESEARCH_FOUNDATION.md` (§13)

**Interfaces:**
- Consumes: `<scratchpad>/refs.md`.
- Produces: the canonical numbered reference list with DOIs.

- [ ] **Step 1: Deduplicate** the registry into a single numbered list; ensure each entry has a DOI or stable URL.

- [ ] **Step 2: Enforce the count** — confirm 50–80 canonical references. If under 50, dispatch a targeted top-up agent for the thinnest section; if over 80, drop the least-relevant.

- [ ] **Step 3: Re-verify a random sample of ~10** references via `WebFetch`; remove any that no longer confirm.

- [ ] **Step 4: Write §13** and replace inline citation markers in §3–§12 with the final numbering.

- [ ] **Step 5: Commit**

```bash
git add docs/RESEARCH_FOUNDATION.md
git commit -m "Assemble verified reference list"
```

---

### Task 7: Propagate references into existing docs

**Files:**
- Modify: `docs/CITATIONS.md`, `docs/methodology.md`, `docs/MODEL_CARD.md`, `docs/CAPABILITIES.md`, `README.md`, `ROADMAP.md`

- [ ] **Step 1: CITATIONS.md** — add subsections (DL segmentation, DVC/DIC, prediction, digital twin, standards) with curated references.

- [ ] **Step 2: methodology.md** — add inline citations in the relevant method sections (structure tensor, Advani–Tucker, U-Net).

- [ ] **Step 3: MODEL_CARD.md** — link segmentation/foundation-model and UQ references.

- [ ] **Step 4: CAPABILITIES.md** — link `RESEARCH_FOUNDATION.md` from the roadmap section.

- [ ] **Step 5: README.md + ROADMAP.md** — add a docs-table row and a roadmap link to `RESEARCH_FOUNDATION.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/ README.md ROADMAP.md
git commit -m "Propagate survey references across documentation"
```

---

### Task 8: Link the survey to the roadmap epics

**Files:** none (GitHub issues — outward-facing; confirm before posting)

- [ ] **Step 1: Post research-basis comments** to epics #18–#21, each citing the relevant `RESEARCH_FOUNDATION.md` section and 3–5 key references:

```bash
gh issue comment 19 --body "Research basis: see docs/RESEARCH_FOUNDATION.md §8 (DVC & DIC). Key refs: <…>"
```

---

### Task 9: Final verification and integration

**Files:** none (verification) — then integrate per maintainer preference

- [ ] **Step 1: Reference count** — confirm 50–80 canonical references in §13.

- [ ] **Step 2: Attribution scan**

```bash
git diff main... | grep -iE "claude|co-authored|anthropic|AI-generated" || echo "clean"
```

- [ ] **Step 3: Link integrity** — confirm every `docs/RESEARCH_FOUNDATION.md` internal link and the new README/ROADMAP links resolve.

- [ ] **Step 4: Spot-check** — re-fetch 3 random references end to end.

- [ ] **Step 5: Integrate** — merge `docs/research-foundation-survey` to `main` and push, or open a PR (maintainer's choice). Confirm CI is green.

---

## Self-Review

**Spec coverage:** §3 standards → Task 2/A1; §4–§5 acquisition/segmentation → Task 2/A2,A3; §6–§7 orientation/tracking → Task 3/A4; §8 DVC/DIC → Task 3/A5; §9 prediction → Task 3/A6; §10–§11 digital twin/cross-cutting → Task 4/A7; §1,§2,§12 → Task 5; §13 references → Task 6; doc propagation → Task 7; issue linkage → Task 8; verification (count/attribution/links/sample) → Task 9. All spec requirements mapped.

**Placeholder scan:** Research prose is generated at execution time by design; every task specifies the exact agent prompt, verification command, target section, and commit — no "TBD"/"handle later" steps.

**Consistency:** Section numbers (§1–§13), agent labels (A1–A7), the shared agent contract, and the 50–80 reference cap are used consistently across tasks and match the spec.
