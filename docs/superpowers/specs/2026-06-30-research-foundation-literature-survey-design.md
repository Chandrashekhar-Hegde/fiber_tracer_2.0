# Research Foundation Literature Survey — Design

Date: 2026-06-30
Status: approved (design); implementation plan to follow

## Context

Fiber Tracer (RAFA) is a regime-aware 3D fiber-analysis toolkit for X-ray CT of
fiber-reinforced composites. Its scholarly grounding today is a short
`CITATIONS.md` and a `methodology.md` citations section. The maintainer wants a
**publication-grade research foundation**: a rigorous, current literature survey
that establishes the academic and research background, is framed for
high-reliability (aerospace/defense) use, and spans the full
**analysis → prediction → digital-twin continuum** rather than analysis alone.

The foundation serves two ends: (1) scholarly credibility that can later support
a paper/whitepaper, and (2) an evidence base that informs the open roadmap epics
(#18 segmentation quality, #19 DVC, #20 DIC, #21 digital twin) before they are
built. This document specifies the survey; a separate implementation plan will
execute it. Building prediction/DVC/DIC/digital-twin *code* is explicitly out of
scope here.

**Constraint:** no Claude/AI attribution in any committed artifact (doc text,
references, commits). See the project memory `no-claude-attribution`.

## Goals

- A new `docs/RESEARCH_FOUNDATION.md`: a **focused, current** survey
  (emphasis 2020–2026 plus seminal works) across 13 themes.
- **~50–80 real, verifiable references** with DOIs, gathered and confirmed via
  live web research; zero fabricated citations.
- Curated references propagated into `CITATIONS.md`, `methodology.md`,
  `MODEL_CARD.md`, and `CAPABILITIES.md`; the new doc linked from `README.md`
  and `ROADMAP.md`.
- A synthesis/gap-analysis section mapping findings to epics #18–#21, with a
  "research basis" comment posted to each epic.

## Non-goals

- No implementation of prediction, DVC, DIC, or digital-twin features.
- Not an exhaustive (150+ reference) review; depth is "focused & current".
- No marketing language; no unverified or fabricated citations.

## Deliverable: `docs/RESEARCH_FOUNDATION.md` structure

1. Scope & motivation (high-reliability aerospace/defense context)
2. Composites & XCT characterization — background
3. **Standards & qualification** — CMH-17; ASTM E1441/E1570/E1695; ISO 15708;
   NADCAP; void/porosity acceptance criteria *(aerospace-grade anchor)*
4. Acquisition & preprocessing (beam hardening, ring/partial-volume artifacts,
   denoising, normalization)
5. Thresholding & segmentation — classical (Otsu/adaptive/multi-Otsu/watershed);
   deep learning (U-Net/nnU-Net); foundation/promptable models
6. Fiber orientation & morphometry (structure tensor, Advani–Tucker A2, PCA,
   diameter/length/tortuosity)
7. Fibre tracking & network analysis (skeletonization, centerline graphs)
8. Deformation measurement — **DVC & DIC** (methods, libraries, strain fields)
9. **Prediction** — micromechanics, ML surrogate models, FE coupling,
   failure/fatigue, uncertainty quantification
10. **Digital twins** for composites (definition, data assimilation, model updating)
11. Cross-cutting — domain adaptation/generalization, validation & benchmarking,
    reproducibility & traceability
12. Synthesis — the analysis→prediction→digital-twin continuum, gap analysis,
    and mapping to epics #18–#21
13. References (full, with DOIs)

## Execution approach: parallel research subagents

Themes are grouped into research subagents dispatched in waves (respecting the
parallel-agent limit). Each agent owns one or two adjacent sections.

Suggested grouping (≈7 agents):
- A1: Standards & qualification (§3)
- A2: Acquisition/preprocessing + classical thresholding/segmentation (§4, §5-classical)
- A3: DL & foundation-model segmentation (§5-DL)
- A4: Orientation, morphometry, fibre tracking (§6, §7)
- A5: DVC & DIC (§8)
- A6: Prediction — micromechanics, ML surrogates, FE, failure, UQ (§9)
- A7: Digital twins + cross-cutting (§10, §11)

**Per-agent contract.** Each agent returns:
- a section draft (≈300–600 words, neutral scholarly tone, no attribution);
- a reference list — for each: authors, year, title, venue, DOI, source URL, and
  a `verified` flag set only if the agent fetched a real page confirming the
  metadata;
- a short list of key open gaps (feeds §12).

Agents must cite **only sources they fetched**. The maintainer synthesizes:
deduplicates references into a single registry, writes the unified document in
one voice, and authors §1, §2, and §12.

## Citation-integrity protocol

- Every reference is confirmed by fetching a real page (publisher, arXiv,
  Crossref, or DOI resolver) that matches authors/title/year/venue/DOI.
- References that cannot be confirmed are **excluded** from the canonical list;
  if useful, they go in an explicit "to verify" appendix, never cited as fact.
- Standards are cited by official designation (e.g., ASTM E1441, ISO 15708).
- After assembly, re-verify a random sample of ~10 references end to end.

## Documentation updates

- `CITATIONS.md` — add new subsections (DL segmentation, DVC/DIC, prediction,
  digital twin, standards) with the curated references.
- `methodology.md` — cite surveyed works inline in the relevant method sections.
- `MODEL_CARD.md` — link segmentation/foundation-model and UQ references.
- `CAPABILITIES.md` — link `RESEARCH_FOUNDATION.md` from the roadmap section.
- `README.md` (docs table) and `ROADMAP.md` — add a link to the new doc.

## Issue linkage

After the doc lands, post a concise "research basis" comment on epics #18–#21,
each pointing to the relevant section and 3–5 key references.

## Tooling

- `WebSearch` / `WebFetch` (deferred tools — load via ToolSearch before use) for
  discovery and verification.
- `Agent` (research subagents) dispatched in waves.

## Verification

- `docs/RESEARCH_FOUNDATION.md` exists with all 13 sections populated.
- Reference count is 50–80; every canonical reference has a DOI/URL and was
  fetched; a re-verified random sample of ~10 passes.
- Curated references appear in the four updated docs; new doc linked from README
  and ROADMAP; all internal links resolve.
- §12 maps to epics #18–#21; "research basis" comments posted.
- No Claude/AI attribution anywhere.

## Risks & mitigations

- **Citation hallucination** → fetch-verify protocol; exclude unverified.
- **Paywalled sources** → cite via DOI/Crossref metadata and abstracts.
- **Scope creep** → reference cap (≤80) and focused depth.
- **Stale/over-broad standards claims** → cite standards by designation only;
  avoid interpreting acceptance thresholds beyond what the standard states.
