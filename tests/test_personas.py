import pytest

from clinicalchat import engine as engine_mod
from clinicalchat import store as store_mod
from clinicalchat.personas import PERSONAS, authenticate, get_persona, signup
from clinicalchat.rbac import Role


@pytest.fixture()
def isolated(engine, store, monkeypatch):
    monkeypatch.setattr(store_mod, "_STORE", store)
    monkeypatch.setattr(engine_mod, "_ENGINE", engine)
    return store


def test_one_persona_per_role_and_patients_bound(engine):
    assert {p.role for p in PERSONAS.values()} == set(Role)
    for p in PERSONAS.values():
        if p.role == Role.PATIENT:
            assert engine.patient(p.patient_key) is not None


def test_authenticate_demo(isolated):
    assert authenticate("dr.sharma", "demo@123")
    assert not authenticate("dr.sharma", "wrong")
    assert not authenticate("nobody", "demo@123")


def test_signup_creates_private_patient_record(isolated, engine):
    ok, key = signup("ananya.new", "s3cret-pass", "s3cret-pass", "ananya rao", "54", "Female", "penicillin", "metformin")
    assert ok
    assert key != "ananya-rao-54"  # never merged into the existing patient with the same name
    p = engine.patient(key)
    assert p.name == "Ananya Rao" and not p.section_ids
    assert [a.allergen for a in p.allergies] and "metformin" in p.medications
    assert authenticate("ananya.new", "s3cret-pass") and not authenticate("ananya.new", "wrong-pass")
    persona = get_persona("ananya.new")
    assert persona.role == Role.PATIENT and persona.patient_key == key
    assert isolated.get_user("ananya.new")["pw_hash"] != "s3cret-pass"  # hashed


@pytest.mark.parametrize("args,msg", [
    (("dr.sharma", "password1", "password1", "X Y", "30", ""), "taken"),
    (("Bad Name", "password1", "password1", "X Y", "30", ""), "Username"),
    (("new.user", "short", "short", "X Y", "30", ""), "8 characters"),
    (("new.user", "password1", "password2", "X Y", "30", ""), "do not match"),
    (("new.user", "password1", "password1", "X Y", "abc", ""), "valid age"),
])
def test_signup_validation(isolated, args, msg):
    ok, err = signup(*args)
    assert not ok and msg in err
