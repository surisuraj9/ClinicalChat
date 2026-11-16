import pytest

from clinicalchat.domain.personas import PERSONAS
from clinicalchat.domain.roles import Role


@pytest.fixture()
def accounts(container):
    return container.accounts


def test_one_persona_per_role_and_patients_bound(kb):
    assert {p.role for p in PERSONAS.values()} == set(Role)
    for p in PERSONAS.values():
        if p.role == Role.PATIENT:
            assert kb.patients.get(p.patient_key) is not None


def test_authenticate_demo(accounts):
    assert accounts.authenticate("dr.sharma", "demo@123")
    assert not accounts.authenticate("dr.sharma", "wrong")
    assert not accounts.authenticate("nobody", "demo@123")


def test_signup_creates_private_patient_record(accounts, repos, kb):
    ok, key = accounts.signup("ananya.new", "s3cret-pass", "s3cret-pass", "ananya rao", "54", "Female", "penicillin", "metformin")
    assert ok
    assert key != "ananya-rao-54"  # never merged into the existing patient with the same name
    p = kb.patients.get(key)
    assert p.name == "Ananya Rao" and not p.section_ids
    assert [a.allergen for a in p.allergies] and "metformin" in p.medications
    assert accounts.authenticate("ananya.new", "s3cret-pass") and not accounts.authenticate("ananya.new", "wrong-pass")
    persona = accounts.get_persona("ananya.new")
    assert persona.role == Role.PATIENT and persona.patient_key == key
    assert repos.users.get("ananya.new")["pw_hash"] != "s3cret-pass"  # hashed


@pytest.mark.parametrize("args,msg", [
    (("dr.sharma", "password1", "password1", "X Y", "30", ""), "taken"),
    (("Bad Name", "password1", "password1", "X Y", "30", ""), "Username"),
    (("new.user", "short", "short", "X Y", "30", ""), "8 characters"),
    (("new.user", "password1", "password2", "X Y", "30", ""), "do not match"),
    (("new.user", "password1", "password1", "X Y", "abc", ""), "valid age"),
])
def test_signup_validation(accounts, args, msg):
    ok, err = accounts.signup(*args)
    assert not ok and msg in err
