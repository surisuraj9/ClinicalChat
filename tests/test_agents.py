from clinicalchat.analysis import ClinicalAnalysisAgent
from clinicalchat.extraction import DataExtractionAgent, assess_document, free_text_labs, parse_upload
from clinicalchat.models import Domain
from clinicalchat.rbac import Role
from clinicalchat.risk import RiskDetectionAgent


def test_extraction_focuses_on_requested_panel(engine):
    ext = DataExtractionAgent(engine).run("Explain my thyroid results", Role.PATIENT, "rahul-mehta-32", [Domain.LAB])
    assert ext.sources == ["Thyroid function test sample zz.docx"] and ext.quality.level == "HIGH"


def test_extraction_reports_missing_data(engine):
    ext = DataExtractionAgent(engine).run("Explain my x-ray", Role.PATIENT, "riya-das-27", [Domain.RADIOLOGY])
    assert any("No radiology reports" in m for m in ext.missing)


def test_radiologist_extraction_never_returns_lab_sections(engine):
    ext = DataExtractionAgent(engine).run("Summarise everything", Role.RADIOLOGIST, "ananya-rao-54", [])
    assert all(engine.store.domain_of(s) != Domain.LAB for s in ext.sections)


def test_free_text_lab_lines():
    rows = free_text_labs("Hemoglobin 9.2 g/dL 13.0-17.0\nWBC: 14500 /uL (ref 4000-11000)\nAge: 35 years 30-40")
    assert [(r.parameter, r.status.value) for r in rows] == [("Hemoglobin", "LOW"), ("WBC", "HIGH")]


def test_upload_name_mismatch_lowers_confidence(tmp_path):
    f = tmp_path / "report.txt"
    f.write_text("Lipid profile\nPatient Name: Somebody Else\nDate: 01/02/2026\nTotal cholesterol 260 mg/dL 0-200\n"
                 "LDL 180 mg/dL 0-130\nHDL 35 mg/dL 40-60\n")
    doc = parse_upload(f, patient_key="me-1", patient_name="Real Person")
    assert doc.domain == Domain.LAB and all(s.patient_key == "me-1" for s in doc.sections)
    q = assess_document(doc, "Real Person")
    assert any("name" in i.lower() for i in q.issues) and q.score < 1


def test_analysis_and_risk_for_abnormal_cbc(engine):
    ext = DataExtractionAgent(engine).run("Interpret the CBC", Role.CLINICIAN, "ananya-rao-54", [Domain.LAB])
    an = ClinicalAnalysisAgent(engine).run(ext, [], [])
    assert an.abnormal and any(c["condition"] for c in an.lab_conditions)
    risk = RiskDetectionAgent(engine.kg).run(patient=engine.patient("ananya-rao-54"), lab_results=an.lab_results, imaging=[],
                                             symptoms=[], proposed_drug_alerts=[], medication_alerts=[],
                                             conditions=an.possible_conditions, inconsistencies=[], low_confidence=False)
    assert risk.risk_level in ("MODERATE", "HIGH") and risk.escalate


def test_clinical_inconsistency_when_symptoms_not_explained(engine):
    ext = DataExtractionAgent(engine).run("I have a rash and diarrhea", Role.PATIENT, "rahul-mehta-32", [Domain.LAB],
                                          new_symptoms=["rash", "diarrhea"])
    an = ClinicalAnalysisAgent(engine).run(ext, [], [])
    assert an.inconsistencies and "not explained" in an.inconsistencies[0]


def test_historical_red_flags_do_not_retrigger_emergency(engine):
    risk = RiskDetectionAgent(engine.kg)
    common = dict(patient=None, lab_results=[], imaging=[], proposed_drug_alerts=[], medication_alerts=[], conditions=[],
                  inconsistencies=[], low_confidence=False)
    assert risk.run(symptoms=["chest pain"], new_symptoms=["chest pain"], **common).risk_level == "CRITICAL"
    old = risk.run(symptoms=["chest pain"], new_symptoms=[], **common)
    assert old.risk_level == "MODERATE" and old.escalate


def test_drug_question_is_not_a_clinical_emergency(engine):
    ext = DataExtractionAgent(engine).run("Is ibuprofen safe for me?", Role.PATIENT, "sneha-gupta-42", [Domain.ALLERGY])
    an = ClinicalAnalysisAgent(engine).run(ext, ["ibuprofen"], [])
    risk = RiskDetectionAgent(engine.kg).run(patient=engine.patient("sneha-gupta-42"), lab_results=an.lab_results, imaging=[],
                                             symptoms=[], proposed_drug_alerts=an.drug_alerts, medication_alerts=[],
                                             conditions=[], inconsistencies=[], low_confidence=False)
    assert risk.escalate and risk.risk_level == "HIGH" and risk.clinical_level != "CRITICAL"
