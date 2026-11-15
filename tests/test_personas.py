from clinicalchat.personas import PERSONAS, authenticate
from clinicalchat.rbac import Role


def test_one_persona_per_role_and_patients_bound(engine):
    assert {p.role for p in PERSONAS.values()} == set(Role)
    for p in PERSONAS.values():
        if p.role == Role.PATIENT:
            assert engine.patient(p.patient_key) is not None


def test_authenticate():
    assert authenticate("dr.sharma", "demo@123")
    assert not authenticate("dr.sharma", "wrong")
    assert not authenticate("nobody", "demo@123")
