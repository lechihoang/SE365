# AI Agent — Implementation Notes

## 1. What Was Implemented

| File | Description |
|---|---|
| `agent/__init__.py` | `ExplanationAgent` class — orchestrates the full pipeline |
| `agent/config.py` | `AgentConfig` — model names, API key resolution, parameters |
| `agent/evidence_loader.py` | `EvidenceLoader` — loads XAI artifacts from disk |
| `agent/evidence_builder.py` | `EvidenceBuilder` — compresses artifacts into text evidence |
| `agent/prompt_builder.py` | `PromptBuilder` — constructs system + user prompts |
| `agent/openai_client.py` | `OpenAIClient` — API wrapper with retry/rate-limit handling |
| `agent/output_schema.py` | JSON schema + `score_to_level()` helper |
| `agent/validator.py` | `OutputValidator` — schema + grounding validation |
| `agent/report_generator.py` | `ReportGenerator` — saves JSON/Markdown/CSV reports |
| `agent/notebooks/AI_Agent_Demo.ipynb` | Interactive Colab-ready demo notebook |

## 2. Proposal Compliance

The implementation follows `AI_agent_proposal.md` sections 1-18. Key correspondences:

| Proposal Section | Implementation |
|---|---|
| Section 5 (Model Choice) | `AgentConfig` with configurable `batch_model` / `report_model` |
| Section 6 (Prompt Architecture) | `PromptBuilder` with system prompt grounding rules |
| Section 7 (Evidence Grounding) | Prompt instructs no hallucination; `OutputValidator` checks SHAP grounding |
| Section 8 (JSON Schema) | `output_schema.py` with `AGENT_OUTPUT_SCHEMA` |
| Section 10 (Evidence Extraction) | `EvidenceLoader` loads from exact codebase artifact paths |
| Section 11 (Compression) | `EvidenceBuilder` uses Top-K for attention/cross-attention/LIME |
| Section 14 (OpenAI Integration) | `OpenAIClient` with retry, timeout, structured JSON output |
| Section 16 (Validation) | `OutputValidator` with schema, score-level, and SHAP checks |

## 3. Deviations from Proposal

### 3.1 Vision mode not implemented

The proposal describes an optional vision mode. The current implementation supports text-only mode. Vision mode is deferred — the model parameter is accepted but images are not sent to the API.

### 3.2 LIME evidence format flexibility

The proposal assumes LIME weights are `[[word, weight], ...]`. The actual format varies between list-of-lists and dict. `EvidenceBuilder._build_lime()` handles both.

### 3.3 No `agent/requirements.txt`

Dependencies (`openai`, `jsonschema`, `python-dotenv`) are installed in the notebook pip cell. A separate requirements file was not created to avoid duplication with the root `requirements.txt`.

## 4. File Structure

```
agent/
├── __init__.py              # ExplanationAgent class
├── config.py                # AgentConfig
├── evidence_loader.py       # Load XAI artifacts
├── evidence_builder.py      # Compress to text evidence
├── prompt_builder.py        # Build OpenAI prompts
├── openai_client.py         # OpenAI API wrapper
├── output_schema.py         # JSON schema
├── report_generator.py      # Save reports
├── validator.py             # Output validation
└── notebooks/
    └── AI_Agent_Demo.ipynb  # Colab-ready demo
```

## 5. OpenAI API Usage

- **Authentication**: API key from `OPENAI_API_KEY` env var, `.env` file, or Colab secrets
- **Structured output**: `response_format={"type": "json_object"}`
- **Retry**: Exponential backoff on `RateLimitError` and `APITimeoutError`
- **Logging**: Token usage logged via Python `logging` module
- **Security**: API key never logged or printed

## 6. Evidence Grounding Strategy

1. **Prompt-level**: System prompt explicitly forbids hallucination
2. **Compression-level**: Only Top-K evidence items are sent (not raw tensors)
3. **Validation-level**: `OutputValidator` checks SHAP percentages against input
4. **Missing evidence**: Loader records `_missing` list; builder outputs "not available" text

## 7. Output Schema

Defined in `output_schema.py`. Key required fields:
- `sample_id`, `language`, `summary`, `scores`, `confidence`

Optional fields:
- `modality_contribution`, `cross_modal_insights`, `method_agreement`
- `limitations`, `recommendations`, `validation_warnings`

## 8. Validation Strategy

| Check | Implementation |
|---|---|
| JSON schema | `jsonschema.validate()` if package available |
| Score levels | Compare `score_to_level(score)` vs claimed level |
| SHAP grounding | Compare claimed `text_origin_pct` vs evidence within 5% tolerance |
| Required fields | Check `summary`, `confidence`, `limitations` are present |

## 9. Security

- API key resolved lazily and never stored in code
- `.gitignore` already excludes `.env` patterns
- Colab secrets integration via `google.colab.userdata`
- `openai_client.py` never logs the API key value

## 10. How to Run

1. Open `agent/notebooks/AI_Agent_Demo.ipynb` in Google Colab
2. Set `OPENAI_API_KEY` in Colab secrets
3. Run all cells
4. Outputs are saved to `{EXP_DIR}/agent_outputs/`

## 11. Known Limitations

1. **Vision mode not implemented** — only text evidence is sent to OpenAI
2. **No streaming** — full response is awaited before processing
3. **LIME format variance** — different LIME versions may produce slightly different JSON structures
4. **Offline mode unavailable** — requires internet access for OpenAI API
5. **Hallucination risk** — despite grounding rules, LLM output should be human-reviewed for thesis
6. **Cost** — each API call costs tokens; all modes use `gpt-4o`

## 12. Output Quality Improvements (V2)

Issues found during real testing and fixes applied:

| # | Problem | Fix | Files Changed |
|---|---|---|---|
| 1 | Schema `level` enum violation — LLM returned Vietnamese display text instead of machine-readable enum | Added `very_poor`/`poor` levels, added `level_display()` helper, report generator renders display text separately | `output_schema.py`, `report_generator.py`, `prompt_builder.py` |
| 2 | Not all 5 targets explained | Schema now requires all 5 (`required: [food, price, atmos, service, overall]`), prompt explicitly demands all 5, validator checks for missing targets | `output_schema.py`, `prompt_builder.py`, `validator.py` |
| 3 | Misleading score wording (7.5 called "trung bình") | Revised score ranges: 0-2 very_poor, 2-4 poor, 4-6 average, 6-8 good, 8-10 excellent | `output_schema.py`, `prompt_builder.py` |
| 4 | Hallucinated explanations (price "reasonable" when review never mentions price) | Strengthened anti-hallucination rules in system prompt — explicit "NEVER invent evidence" instruction with examples | `prompt_builder.py` |
| 5 | Cross-attention too generic | Prompt now instructs to reference actual token names, patch coordinates (row,col), and attention scores | `prompt_builder.py` |
| 6 | SHAP percentages missing target context | Prompt requires per-target SHAP breakdown, schema adds `per_target` field in `modality_contribution` | `prompt_builder.py`, `output_schema.py` |
| 7 | Weak limitations section | Prompt requires at least 3 meaningful limitations with specific required items, validator checks count | `prompt_builder.py`, `validator.py` |
| 8 | No evidence completeness section | Added `evidence_completeness` field (gradcam/attention/cross_attention/shap/lime booleans + total), report generator renders it, `__init__.py` overrides with ground truth from evidence loader | `output_schema.py`, `__init__.py`, `report_generator.py`, `validator.py` |
| 9 | Proposal compliance gaps | Added missing fields: `per_target` SHAP, `confidence_reasoning`, `evidence_completeness` | All schema/prompt/validator files |
| 10 | Generic recommendations | Prompt now instructs: only suggest improvements directly supported by evidence, never recommend changes for topics not mentioned in review | `prompt_builder.py` |
| 11 | Report sections incomplete | Markdown report now follows fixed 13-section order: Review → Predictions → GT → Summary → Scores (all 5) → Evidence Completeness → SHAP → Cross-Attention → Method Agreement → Limitations → Recommendations → Confidence → Warnings | `report_generator.py` |
| 12 | No cross-method agreement section | Added `method_agreement` as required field, validator checks presence, report renders it | `prompt_builder.py`, `validator.py`, `report_generator.py` |
| 13 | Arbitrary confidence | Prompt defines clear rules (high=4-5 methods+agreement, medium=2-3, low=0-1), added `confidence_reasoning` field, validator checks for its presence | `prompt_builder.py`, `output_schema.py`, `validator.py` |
| 14 | Awkward Vietnamese translations of technical terms | Prompt explicitly lists English terms to keep: Grad-CAM, Cross-Attention, SHAP, LIME, text-origin, image-origin, token, patch | `prompt_builder.py` |

## 13. OpenAI Model Migration

The project migrated from a two-tier model strategy (`gpt-4o-mini` for batch, `gpt-4o` for reports) to **`gpt-4o` exclusively**.

**Reason:** `gpt-4o-mini` produced lower-quality explanations — hallucinated evidence, missed targets, inconsistent score wording. The quality improvement from `gpt-4o` justifies the cost increase.

**Files updated:**
- `agent/config.py` — all three defaults (`batch_model`, `report_model`, `vision_model`) set to `gpt-4o`
- `agent/__init__.py` — docstring example updated
- `agent/notebooks/AI_Agent_Demo.ipynb` — config cell updated, model name printed before first API call

**Validation:** No active runtime path references `gpt-4o-mini`. The proposal document (`AI_agent_proposal.md`) retains historical references to the two-tier strategy for context.

---

## 14. Future Improvements

- Vision mode: send Grad-CAM overlays to `gpt-4o` for richer visual description
- Streaming: support streaming responses for interactive notebooks
- Caching: cache API responses to avoid re-calling for the same sample
- RAG: retrieve similar historical reviews for contextual comparison
- Offline fallback: template-based explanation when API is unavailable
