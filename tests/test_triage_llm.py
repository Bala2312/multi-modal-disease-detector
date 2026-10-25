import pytest

from triage_llm import (
    ClinicalAssistant,
    TriageLevel,
    determine_triage,
)


def test_emergency_rule_takes_priority_over_standard_consult() -> None:
    result = determine_triage({"Pneumothorax": 0.76, "Pneumonia": 0.91})

    assert result.level is TriageLevel.EMERGENCY
    assert result.trigger_label == "Pneumothorax"
    assert result.confidence == 0.76


def test_unlisted_positive_disease_routes_to_standard_consult() -> None:
    result = determine_triage({"Effusion": 0.63})

    assert result.level is TriageLevel.STANDARD_CONSULT
    assert result.trigger_label == "Effusion"


def test_normal_label_routes_to_low_risk_when_no_abnormal_trigger() -> None:
    result = determine_triage({"No Finding": 0.89, "Pneumonia": 0.12})

    assert result.level is TriageLevel.LOW_RISK
    assert result.confidence == 0.89


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        determine_triage({"Pneumonia": 1.2})


def test_no_rag_context_does_not_call_llm() -> None:
    triage = determine_triage({"Pneumonia": 0.6})
    assistant = ClinicalAssistant.__new__(ClinicalAssistant)

    summary = assistant.summarize("Pneumonia", 0.6, triage, [])

    assert "No relevant guideline passages" in summary
