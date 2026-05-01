"""Tests for individual adapters — focused on the ones with no vendor deps."""

import pytest
from harness.adapters.calculator_adapter import CalculatorAdapter
from harness.adapters.phenoage_adapter import PhenoAgeAdapter
from harness.adapters.wearable_adapter import WearableAdapter
from harness.adapters.ncbi_tools_adapter import NCBIToolsAdapter


class TestCalculatorAdapter:
    @pytest.fixture
    def adapter(self):
        return CalculatorAdapter()

    @pytest.mark.asyncio
    async def test_cha2ds2_vasc(self, adapter):
        result = await adapter.run(
            "Calculate stroke risk",
            context={
                "calculator": "cha2ds2_vasc",
                "calculator_params": {
                    "age": 72,
                    "sex": "female",
                    "hypertension": True,
                    "stroke_tia_history": True,
                },
            },
        )
        assert result["source"] == "clinical_calculators"
        assert result["confidence"] == 0.95
        assert "Score" in result["answer"]

    @pytest.mark.asyncio
    async def test_bmi(self, adapter):
        result = await adapter.run(
            "Calculate BMI",
            context={
                "calculator": "bmi",
                "calculator_params": {"weight_kg": 80, "height_m": 1.75},
            },
        )
        assert "26.1" in result["answer"]

    @pytest.mark.asyncio
    async def test_unknown_calculator(self, adapter):
        result = await adapter.run(
            "Calculate something",
            context={"calculator": "nonexistent", "calculator_params": {}},
        )
        assert "Available" in result["answer"]

    @pytest.mark.asyncio
    async def test_no_calculator_specified(self, adapter):
        result = await adapter.run("What calculators are available?")
        assert "Available" in result["answer"]
        assert result["confidence"] == 0.3


class TestPhenoAgeAdapter:
    @pytest.fixture
    def adapter(self):
        return PhenoAgeAdapter()

    @pytest.mark.asyncio
    async def test_with_full_biomarkers(self, adapter):
        result = await adapter.run(
            "Calculate biological age",
            context={
                "age": 50,
                "biomarkers": {
                    "albumin": 4.0,
                    "creatinine": 1.0,
                    "glucose": 100,
                    "crp": 0.1,
                    "lymphocyte_pct": 30,
                    "mcv": 90,
                    "rdw": 13,
                    "alkaline_phosphatase": 60,
                    "wbc": 6.0,
                },
            },
        )
        assert result["source"] == "phenoage"
        assert "PhenoAge" in result["answer"]
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_missing_biomarkers(self, adapter):
        result = await adapter.run("Calculate biological age", context={"age": 50})
        assert "Missing" in result["answer"] or "requires" in result["answer"]
        assert result["confidence"] == 0.1


class TestWearableAdapter:
    @pytest.fixture
    def adapter(self):
        return WearableAdapter()

    @pytest.mark.asyncio
    async def test_with_activity_data(self, adapter):
        result = await adapter.run(
            "Analyse activity",
            context={
                "wearable_data": {
                    "avg_steps_30d": 8000,
                    "trend": "stable",
                },
            },
        )
        assert result["source"] == "wearable"
        assert "Health Score" in result["answer"]

    @pytest.mark.asyncio
    async def test_no_data(self, adapter):
        result = await adapter.run("Analyse health")
        assert "No wearable data" in result["answer"]


class TestNCBIToolsAdapter:
    @pytest.fixture
    def adapter(self):
        return NCBIToolsAdapter()

    def test_capabilities(self, adapter):
        caps = adapter.capabilities()
        assert "gene_lookup" in caps
        assert "pubmed_search" in caps

    def test_available(self, adapter):
        assert adapter.available is True
        assert adapter.modality == "genomics"
