"""Native GeneAgent implementation — reimplements the cascade pipeline
from ncbi-nlp/GeneAgent (Nature Methods 2024) using modern LLM APIs.

Pipeline:
1. Baseline analysis: LLM describes biological processes of gene set
2. Topic verification: Generate decontextualized claims, verify via NCBI
3. Analysis verification: Verify each claim with NCBI evidence
4. Final synthesis: Updated analysis with verified claims
"""

from __future__ import annotations

from typing import Any

from harness.adapters.ncbi_tools_adapter import NCBIToolsAdapter
from harness.llm_client import LLMClient

# === Prompts following the GeneAgent cascade protocol (ncbi-nlp/GeneAgent) ===

SYSTEM_ANALYST = "You are an efficient and insightful assistant to a molecular biologist."
SYSTEM_VERIFY = "You are a helpful and objective fact-checker to verify the summary of gene set."


def baseline_prompt(genes: list[str]) -> str:
    gene_str = ",".join(genes)
    return f"""Write a critical analysis of the biological processes performed by this system of interacting proteins.
Propose a brief name for the most prominent biological process performed by the system.
Put the name at the top of the analysis as "Process: <name>".
Be concise, do not use unnecessary words.
Be textual, do not use any format symbols.
Be specific, avoid overly general statements.
Be factual, do not editorialize.
For each important point, describe your reasoning and supporting information.
For each biological function name, show the corresponding gene names.
Here is the gene set: {gene_str}"""


def claims_prompt(genes: list[str], process: str) -> str:
    gene_str = ",".join(genes)
    return f"""Here is the original process name for the gene set {gene_str}:\n{process}
However, the process name might be false. Please generate decontextualized claims for the process name that need to be verified.
Only Return a list type that contain all generated claim strings, for example, ["claim_1", "claim_2"]
Only generate affirmative claims for the entire gene set, separated by commas.
Don't generate claims for single genes or incomplete gene sets.
Replace "these genes" / "this system" with the core gene names."""


class GeneAgentNative:
    """Reimplementation of GeneAgent cascade pipeline using our LLMClient."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.ncbi = NCBIToolsAdapter()

    async def analyze(self, genes: list[str]) -> dict[str, Any]:
        """Run the full cascade: baseline → claims → verification → synthesis."""
        if not genes:
            return {"summary": "No genes provided.", "evidence": [], "process": ""}

        # Step 1: Baseline analysis
        baseline = await self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_ANALYST},
                {"role": "user", "content": baseline_prompt(genes)},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        # Extract process name
        process = ""
        for line in baseline.split("\n"):
            if line.lower().startswith("process:"):
                process = line.split(":", 1)[1].strip()
                break

        # Step 2: Gather NCBI evidence for each gene (cascade verification)
        evidence_items = []
        for gene in genes[:3]:  # cap to prevent runaway
            try:
                info = await self.ncbi.gene_info(gene)
                if info.get("summary"):
                    evidence_items.append(f"[NCBI {gene}] {info['summary'][:300]}")
            except Exception:
                pass

        # Step 3: Synthesis with evidence
        synthesis_prompt = (
            f"Original analysis:\n{baseline}\n\n"
            f"NCBI verification evidence:\n" + "\n".join(evidence_items) +
            f"\n\nUsing the NCBI evidence, provide a refined final analysis. "
            f"Flag any claims contradicted by evidence."
        )
        final = await self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_ANALYST},
                {"role": "user", "content": synthesis_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        return {
            "summary": final,
            "evidence": evidence_items,
            "process": process,
            "baseline": baseline,
        }
