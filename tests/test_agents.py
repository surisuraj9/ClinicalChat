from clinicalchat.agents.analysis import ClinicalAnalysisAgent
from clinicalchat.agents.extraction import DataExtractionAgent
from clinicalchat.agents.risk import RiskDetectionAgent, assessment, symptom_indicators
from clinicalchat.domain.models import Domain
from clinicalchat.domain.roles import Role
from clinicalchat.ingestion.quality import assess_document
from clinicalchat.ingestion.uploads import free_text_labs, parse_upload


def _extraction(kb):
    return DataExtractionAgent(kb.catalog, kb.patients, kb.kg, kb.records)


def _analysis(kb):
    return ClinicalAnalysisAgent(kb.catalog, kb.patients, kb.kg, kb.safety)


def test_extraction_focuses_on_requested_panel(kb):
    ext = _extraction(kb).run("Explain my thyroid results", Role.PATIENT, "rahul-mehta-32", [Domain.LAB])
    assert ext.sources == ["Thyroid function test sample zz.docx"] and ext.quality.level == "HIGH"


def test_extraction_reports_missing_data(kb):
    ext = _extraction(kb).run("Explain my x-ray", Role.PATIENT, "riya-das-27", [Domain.RADIOLOGY])
    assert any("No radiology reports" in m for m in ext.missing)


def test_radiologist_extraction_never_returns_lab_sections(kb):
    ext = _extraction(kb).run("Summarise everything", Role.RADIOLOGIST, "ananya-rao-54", [])
    assert all(kb.catalog.domain_of(s) != Domain.LAB for s in ext.sections)


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


def test_analysis_and_risk_for_abnormal_cbc(kb):
    ext = _extraction(kb).run("Interpret the CBC", Role.CLINICIAN, "ananya-rao-54", [Domain.LAB])
    an = _analysis(kb).run(ext, [], [])
    assert an.abnormal and any(c["condition"] for c in an.lab_conditions)
    risk = RiskDetectionAgent().run(kb.patients.get("ananya-rao-54"), ext, an)
    assert risk.risk_level in ("MODERATE", "HIGH") and risk.escalate


def test_clinical_inconsistency_when_symptoms_not_explained(kb):
    ext = _extraction(kb).run("I have a rash and diarrhea", Role.PATIENT, "rahul-mehta-32", [Domain.LAB],
                                          new_symptoms=["rash", "diarrhea"])
    an = _analysis(kb).run(ext, [], [])
    assert an.inconsistencies and "not explained" in an.inconsistencies[0]


def test_historical_red_flags_do_not_retrigger_emergency():
    assert assessment(symptom_indicators(["chest pain"], ["chest pain"])).risk_level == "CRITICAL"
    old = assessment(symptom_indicators(["chest pain"], []))
    assert old.risk_level == "MODERATE" and old.escalate


def test_drug_question_is_not_a_clinical_emergency(kb):
    ext = _extraction(kb).run("Is ibuprofen safe for me?", Role.PATIENT, "sneha-gupta-42", [Domain.ALLERGY])
    an = _analysis(kb).run(ext, ["ibuprofen"], [])
    risk = RiskDetectionAgent().run(kb.patients.get("sneha-gupta-42"), ext, an)
    assert risk.escalate and risk.risk_level == "HIGH" and risk.clinical_level != "CRITICAL"


TSH_REPORT = """VIJAYA DIAGNOSTIC CENTRE
THYROIDPROFILE
Name | :Mrs. G Sai Sowmya Sree | Registered on: 16-Nov-2026 09:37
Age/Gender | : 31 Years / Female | Collected on :16-Nov-2026 09:39
Ref. By | : Dr Self | Printed on | : 16-Nov-2026 11:31
Sample Type | : Serum | Regn Centre : Madinaguda
TEST NAME | RESULT | UNIT | BIOLOGICALREFERENCEINTERVAL
T3 Total | 1.54 | ng/mL | Non pregnant | :0.60-1.81
T4 Total | 12.30 | μg/dL | Adult | :3.2-12.6
TSH - Ultrasensitive | 11.225 | μIU/mL | Non pregnant : 0.55 - 4.78
(CLIA) | 1st Trimester | : 0.3 - 4.5
"""


def test_lab_table_found_below_report_banner(tmp_path):
    """Header row below name/age/sample rows; '1st Trimester' is a label, not a value of 1."""
    f = tmp_path / "16thDecTSH.txt"
    f.write_text(TSH_REPORT)
    doc = parse_upload(f, patient_key="p-1", patient_name="G Sai Sowmya Sree")
    labs = [(r.parameter, r.value, r.status.value) for s in doc.sections for r in s.lab_results]
    assert doc.domain == Domain.LAB
    assert labs == [("T3 Total", 1.54, "NORMAL"), ("T4 Total", 12.3, "NORMAL"),
                    ("TSH - Ultrasensitive", 11.225, "HIGH")]


def test_footnote_mentioning_normal_range_is_not_a_table_header():
    from clinicalchat.ingestion.lab_tables import parse_pipe_lines
    lines = ["Pred | 3.16 | 2.72 | * Indicates value outside normal range or significant post change.",
             "#0476 | 162 cm | 79 kg | Female"]
    assert parse_pipe_lines(lines) == []


def _generic_upload(text: str, filename: str = "16thDecTSH.png"):
    from clinicalchat.domain.models import ClinicalDocument, Section
    return ClinicalDocument(doc_id="UPLOAD-x", filename=filename, relative_path=f"uploads/p-1/{filename}",
                            category="patient_record", domain=Domain.PATIENT_RECORD, title="16thDecTSH",
                            origin="upload", sections=[Section(section_id="UPLOAD-x-S00", doc_id="UPLOAD-x",
                                                               title="16thDecTSH", text=text, patient_key="p-1")])


def test_stored_generic_upload_is_reread_as_lab_report():
    from clinicalchat.ingestion.uploads import reparse_generic_upload
    doc = _generic_upload(TSH_REPORT)
    assert reparse_generic_upload(doc) and doc.domain == Domain.LAB
    assert len(doc.sections[0].lab_results) == 3
    note = _generic_upload("Discharge summary. Patient stable, advised rest.", "note.pdf")
    assert not reparse_generic_upload(note) and note.domain == Domain.PATIENT_RECORD


class _Catalog:
    def __init__(self, docs):
        self.all = docs
        self.docs = {s.section_id: d for d in docs for s in d.sections}

    def patient_sections(self, key, domains):
        return [s for d in self.all if d.domain in domains for s in d.sections]

    def domain_of(self, s):
        return self.docs[s.section_id].domain

    def source_of(self, s):
        return self.docs[s.section_id].filename

    def document_of(self, s):
        return self.docs[s.section_id]


def test_lab_question_uses_matching_content_from_other_records(kb):
    """No lab-type report on file, but a clinical note carries the TSH value: use it rather than 'no thyroid data'."""
    note = _generic_upload("Clinic note: TSH 11.2 uIU/mL last week, on no thyroid medicine.", "clinic-note.pdf")
    xray = _generic_upload("Chest X-ray: clear lung fields.", "xray.pdf")
    xray.doc_id = xray.sections[0].doc_id = "UPLOAD-y"
    xray.sections[0].section_id = "UPLOAD-y-S00"
    agent = DataExtractionAgent(_Catalog([note, xray]), kb.patients, kb.kg, kb.records)
    ext = agent.run("What medicine does she need to control the thyroid?", Role.CLINICIAN, "p-1", [Domain.LAB])
    assert ext.sources == ["clinic-note.pdf"]
    assert any("using matching content from other records" in m for m in ext.missing)


def test_thyroid_panel_matched_by_test_names_not_only_title(kb):
    from clinicalchat.ingestion.uploads import reparse_generic_upload
    tsh = _generic_upload(TSH_REPORT, "16thDecTSH.png")
    reparse_generic_upload(tsh)
    agent = DataExtractionAgent(_Catalog([tsh]), kb.patients, kb.kg, kb.records)
    assert agent._focus("is her thyroid controlled?", tsh.sections) == tsh.sections
