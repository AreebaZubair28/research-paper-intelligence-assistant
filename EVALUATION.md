# Manual Evaluation Log

This project includes an LLM-based grounding check (`check_grounding()` in `src/generate.py`)
that verifies whether a generated answer's claims are actually supported by the retrieved
passages, rather than relying only on the model's own self-reported confidence.

Because this component's output isn't deterministic (LLM responses vary in wording run to
run), it isn't covered by automated `pytest` tests like the chunking logic is. Instead, this
file documents real test cases and their results, so the behavior can be manually reviewed
or re-run at any time.

---

## Test Case 1: Answerable question — should be grounded correctly

**Documents loaded:** "Attention Is All You Need" (Transformer paper), "Deep Residual
Learning for Image Recognition" (ResNet paper)

**Question:** "What problem does the degradation problem refer to, and how do residual
connections address it?"

**Expected behavior:** Retrieval should pull relevant passages specifically from the ResNet
paper (this question has nothing to do with the Transformer paper), and the answer should be
fully supported by those passages.

**Actual result:**
- All 3 retrieved sources correctly came from the ResNet paper only — no cross-contamination
  from the unrelated Transformer paper.
- Answer accurately described the degradation problem (accuracy saturating/degrading with
  depth, not caused by overfitting) and how residual connections address it (F(x) := H(x) - x
  reformulation, shortcut connections).
- Self-reported confidence: "Fully supported."
- Independent grounding check: confirmed all claims were supported by the retrieved passages.

**Conclusion:** ✅ Correct — system retrieved the right document, generated an accurate
grounded answer, and both confidence layers agreed it was well-supported.

---

## Test Case 2: Unanswerable question — should be correctly flagged as unsupported

**Documents loaded:** Same two papers as above (neither covers Vision Transformers).

**Question:** "What optimizer learning rate schedule works best for training Vision
Transformers?"

**Expected behavior:** Since neither paper discusses Vision Transformers, the system should
recognize the retrieved passages don't actually answer the question, rather than confidently
fabricating a plausible-sounding answer from general training knowledge.

**Actual result:**
- Retrieval returned the closest available passages (learning rate schedules from the
  original Transformer paper and from ResNet training), since these were the nearest
  semantic matches — but neither actually addresses Vision Transformers.
- The generated answer explicitly stated it could not determine an answer to the question
  from the provided passages, and accurately summarized what each passage *does* cover
  instead of overclaiming.
- Self-reported confidence: "Not supported."
- Independent grounding check: confirmed the "cannot determine" response was accurate, and
  verified none of the passages mention Vision Transformers.

**Conclusion:** ✅ Correct — system did not hallucinate an answer from general knowledge,
and both confidence layers correctly identified the question as unanswerable from the
available evidence.

---

## Why this matters

These two contrasting cases demonstrate the core goal of this project: the system behaves
*differently and correctly* depending on whether genuine supporting evidence exists,
rather than always answering confidently regardless of whether retrieval actually found
relevant content. This is the practical difference between an evidence-grounded system and
a plain LLM call with no retrieval or verification step.

## How to re-run these tests

1. Ensure both PDFs are in `data/`
2. Run `main.py` with the question set to Test Case 1's question, confirm sources and
   confidence match the description above
3. Change the question to Test Case 2's question, confirm the system flags it as
   unsupported rather than answering confidently
