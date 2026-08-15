"""
LLM service used for semantic enrichment.

The LLM is NOT responsible for creating the workflow.

Its job is only to interpret the user's natural-language request and
identify:

1. Explicit actions/concepts
2. Implied actions/concepts
3. Ordering/dependencies
4. Relevant domain concepts

The domain graph remains responsible for grounding those concepts.
"""

import json
import os
import re
from typing import Any, Dict, List


class LLMService:

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        token: str | None = None,
    ):
        self.model = model
        self.token = token or os.getenv("HF_TOKEN")

        self.client = None

        if self.token:
            try:
                from huggingface_hub import InferenceClient

                self.client = InferenceClient(
                    model=self.model,
                    token=self.token,
                )

            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for LLM inference. "
                    "Install it with: pip install huggingface_hub"
                )

    def call_llm(self, prompt: str) -> str:
        """
        Call the actual LLM.

        If HF_TOKEN is not configured, we deliberately fail loudly rather
        than silently pretending that an LLM was used.
        """

        if self.client is None:
            raise RuntimeError(
                "LLM is not configured. Set the HF_TOKEN environment "
                "variable before using LLM semantic enrichment."
            )

        response = self.client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a workflow semantic analysis engine. "
                        "You do NOT execute workflows and you do NOT invent "
                        "API functions. "
                        "Your job is to identify explicit and implicitly "
                        "required domain actions from a user's request."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        return response.choices[0].message.content

    def enrich_prompt(
        self,
        user_prompt: str,
        domain_context: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:

        domain_context = domain_context or []

        context_text = json.dumps(
            domain_context,
            indent=2,
            ensure_ascii=False,
        )

        prompt = f"""
Analyze the following workflow request.

USER REQUEST:
{user_prompt}

AVAILABLE DOMAIN CONTEXT:
{context_text}

Your task is to identify the semantic workflow steps.

IMPORTANT:

1. Preserve actions explicitly mentioned by the user.
2. Infer actions that are normally required to perform the requested operation.
3. Inferred actions are allowed even if no registered function currently exists.
4. Use the available domain context when deciding what actions are relevant.
5. Do NOT invent unrelated actions.
6. Do NOT choose concrete implementation functions.
7. Do NOT create code.
8. Preserve the logical order between actions.
9. If an action is a prerequisite for another action, represent that dependency.
10. Return ONLY valid JSON.

Return this structure:

{{
    "intent": "...",
    "explicit_steps": [
        {{
            "text": "...",
            "reason": "..."
        }}
    ],
    "inferred_steps": [
        {{
            "text": "...",
            "reason": "..."
        }}
    ],
    "dependencies": [
        {{
            "before": "...",
            "after": "..."
        }}
    ]
}}

Example:

User request:
"Transfer funds"

Possible interpretation:

{{
    "intent": "transfer funds",
    "explicit_steps": [
        {{
            "text": "transfer funds",
            "reason": "Explicitly requested by the user."
        }}
    ],
    "inferred_steps": [
        {{
            "text": "validate transfer request",
            "reason": "The transfer request should be validated before processing."
        }},
        {{
            "text": "check account balance",
            "reason": "Funds availability must normally be established before transfer."
        }},
        {{
            "text": "process transfer",
            "reason": "The requested transfer must be executed."
        }},
        {{
            "text": "generate transfer response",
            "reason": "The workflow should produce a result."
        }}
    ],
    "dependencies": [
        {{
            "before": "validate transfer request",
            "after": "check account balance"
        }},
        {{
            "before": "check account balance",
            "after": "process transfer"
        }},
        {{
            "before": "process transfer",
            "after": "generate transfer response"
        }}
    ]
}}
"""

        raw = self.call_llm(prompt)

        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:

        raw = raw.strip()

        # Remove markdown fences if the model returned them.
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )

        try:
            return json.loads(raw)

        except json.JSONDecodeError as exc:

            # Attempt to recover the JSON object from surrounding text.
            start = raw.find("{")
            end = raw.rfind("}")

            if start >= 0 and end > start:

                candidate = raw[start:end + 1]

                try:
                    return json.loads(candidate)

                except json.JSONDecodeError:
                    pass

            raise ValueError(
                f"LLM returned invalid JSON:\n{raw}"
            ) from exc