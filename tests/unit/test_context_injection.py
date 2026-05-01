"""Unit tests for harness.context_injection.format_task_prompt."""

from harness.context_injection import format_task_prompt


class TestContextInjection:
    def test_labbench2_task_injects_context(self):
        task = {
            "question": "What antibiotic was used?",
            "context": {
                "source": "futurehouse/labbench2",
                "sources": ["Smith et al. 2024, Nature"],
                "key_passage": "Cultures were treated with ampicillin at 100 μg/mL.",
            },
        }
        prompt = format_task_prompt(task)
        assert "What antibiotic was used?" in prompt
        assert "ampicillin" in prompt.lower()
        assert "Smith et al." in prompt
        assert "Key Passage" in prompt

    def test_labbench2_via_field_detection(self):
        """Should also work when 'source' key is absent but fields match."""
        task = {
            "question": "What compound?",
            "context": {
                "sources": ["Paper X"],
                "key_passage": "We used compound Y.",
            },
        }
        prompt = format_task_prompt(task)
        assert "compound Y" in prompt

    def test_labbench2_via_benchmark_tag(self):
        """Loader sets context.benchmark='labbench2' — must be recognised."""
        task = {
            "question": "What reagent?",
            "context": {
                "benchmark": "labbench2",
                "subset": "litqa3",
                "sources": ["Doe 2023"],
                "key_passage": "Reagent Z was employed.",
            },
        }
        prompt = format_task_prompt(task)
        assert "Reagent Z" in prompt
        assert "Doe 2023" in prompt

    def test_non_labbench2_preserves_question(self):
        task = {
            "question": "Compute CKD-EPI eGFR.",
            "context": {"subject": "MedCalc"},
        }
        assert format_task_prompt(task) == "Compute CKD-EPI eGFR."

    def test_medcalc_task_question_only(self):
        task = {
            "question": "Patient age 60, Cr 1.2...",
            "context": {},
        }
        assert format_task_prompt(task) == "Patient age 60, Cr 1.2..."

    def test_no_context_field(self):
        task = {"question": "Test?"}
        assert format_task_prompt(task) == "Test?"

    def test_empty_sources_and_passage(self):
        task = {
            "question": "Q?",
            "context": {
                "source": "futurehouse/labbench2",
                "sources": [],
                "key_passage": "",
            },
        }
        # Should just return question since no actual content
        assert format_task_prompt(task).strip() == "Q?"

    def test_sources_only_no_passage(self):
        task = {
            "question": "Q?",
            "context": {"sources": ["Paper 1", "Paper 2"]},
        }
        prompt = format_task_prompt(task)
        assert "Paper 1" in prompt
        assert "Paper 2" in prompt
        assert "Key Passage" not in prompt  # not present

    def test_context_not_dict(self):
        """Graceful handling of unexpected context types."""
        task = {"question": "Q?", "context": "some string"}
        assert format_task_prompt(task) == "Q?"
