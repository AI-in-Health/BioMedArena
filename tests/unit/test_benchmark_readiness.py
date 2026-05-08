from harness.eval.benchmark_readiness import readiness_for_benchmark, registered_benchmark_metadata


def test_readiness_marks_mteb_as_official_ready_candidate_reranking():
    label, reason = readiness_for_benchmark("hf_mteb_medical_retrieval")

    assert label == "official_ready"
    assert "ranked doc ids" in reason


def test_readiness_marks_moleculenet_as_official_ready():
    meta = registered_benchmark_metadata("hf_moleculenet_bbbp")

    assert meta["readiness"] == "official_ready"
    assert meta["official_split"]


def test_readiness_marks_external_agent_benchmarks():
    label, reason = readiness_for_benchmark("medagentbench")

    assert label == "needs_token_or_external_env"
    assert "FHIR" in reason


def test_readiness_marks_vision_and_official_runner_gaps():
    vision_label, vision_reason = readiness_for_benchmark("pathvqa")
    runner_label, runner_reason = readiness_for_benchmark("bixbench")

    assert vision_label == "vision_ready_needs_vision_backend"
    assert "vision-capable" in vision_reason
    assert runner_label == "needs_token_or_external_env"
    assert "notebook" in runner_reason
