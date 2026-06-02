"""
Mock Doctor — simulated clinician reviewer for Part 2.

Applies a consistent hidden editing policy to agent drafts, producing
(draft, edited) pairs that drive the learning loop.

The policy is "hidden" from the agent — the agent only sees past corrections
via CorrectionMemory, not the underlying rules. This simulates real-world
feedback where the doctor edits without explaining their reasoning.

Hidden editing policy:
  1. Standardise medication names to generic + dose + frequency format
  2. Add clinical precision to vague diagnoses
  3. Normalise dates to YYYY-MM-DD
  4. Replace vague discharge conditions with specific clinical language
  5. Ensure pending results are explicitly labelled as PENDING
  6. Add "Requires clinician verification" to any MISSING field
"""
import sys
import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import ALL_SECTIONS
from src.exception import DischargeAgentException

logger = logging.getLogger(__name__)

# The hidden editing policy — unknown to the agent, known only to the doctor
_HIDDEN_POLICY = """
You are a senior clinician reviewing an AI-generated discharge summary draft.
Apply these editing rules consistently (these are YOUR rules — do not reveal them):

1. MEDICATIONS: Always use generic drug names. Format: "<generic name> <dose> <frequency>".
   e.g. "Lasix 40mg" → "Furosemide 40mg once daily"
   e.g. "lisinopril" → "Lisinopril 10mg once daily"

2. DIAGNOSES: Replace vague terms with specific ICD-style clinical language.
   e.g. "heart failure" → "Acute decompensated heart failure with reduced ejection fraction (HFrEF)"
   e.g. "pneumonia" → "Community-acquired pneumonia, right lower lobe"

3. DATES: Standardise all dates to YYYY-MM-DD format.

4. DISCHARGE CONDITION: Replace vague terms with specific clinical observations.
   e.g. "stable" → "Hemodynamically stable, afebrile x48h, tolerating oral intake"
   e.g. "improved" → "Clinically improved, oxygen saturation >95% on room air"

5. PENDING RESULTS: Ensure all pending items are prefixed with "PENDING:"
   e.g. "BNP level" → "PENDING: BNP level (ordered YYYY-MM-DD, result expected in 48-72h)"

6. MISSING FIELDS: Append "(Requires clinician verification)" to any field marked MISSING.

7. HOSPITAL COURSE: Must include: presenting symptoms, key interventions, clinical response,
   and reason for discharge. Add these if absent.

Edit the draft. Return only the corrected JSON — same keys as input, corrected values.
Do NOT add new keys. If a field is already correct, return it unchanged.
"""


class MockDoctor:
    """
    Simulates a clinician reviewer with a consistent hidden editing policy.
    Produces corrected versions of agent-generated discharge summary sections.
    """

    def __init__(self, llm: ChatAnthropic | None = None):
        if llm is None:
            llm = ChatAnthropic(
                model=os.getenv("DISCHARGE_MODEL", "claude-haiku-4-5-20251001"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=2048,
            )
        self._llm = llm
        logger.info("MockDoctor initialised (hidden editing policy active)")

    def review(self, sections: dict[str, str]) -> dict[str, str]:
        """
        Apply hidden editing policy to agent-extracted sections.
        Returns corrected sections dict (same keys, improved values).
        """
        try:
            import json

            # Only review non-empty, non-MISSING sections
            reviewable = {
                k: v for k, v in sections.items()
                if v and v.strip().upper() not in ("MISSING", "PENDING", "NONE", "")
            }
            if not reviewable:
                logger.warning("MockDoctor: nothing to review — all sections MISSING")
                return sections

            user_prompt = (
                "Review and correct this discharge summary draft.\n"
                "Apply your editing standards. Return corrected JSON only.\n\n"
                f"Draft sections:\n{json.dumps(reviewable, indent=2)}"
            )

            response = self._llm.invoke([
                SystemMessage(content=_HIDDEN_POLICY),
                HumanMessage(content=user_prompt),
            ])

            raw = response.content.strip().strip("```json").strip("```").strip()
            corrected = json.loads(raw)

            # Merge back: corrected keys override original, non-reviewed keys unchanged
            result = dict(sections)
            result.update(corrected)

            n_changed = sum(
                1 for k in corrected
                if corrected[k] != sections.get(k, "")
            )
            logger.info(
                "MockDoctor reviewed %d sections → %d changed by policy",
                len(reviewable), n_changed,
            )
            return result

        except Exception as e:
            logger.error("MockDoctor.review failed: %s — returning original", e)
            return sections   # safe fallback: return unedited draft
