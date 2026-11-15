"""Self-registration page for new patients (plain HTML form served by FastAPI, linked from the Gradio login)."""

from __future__ import annotations

import html
from typing import Dict, Optional

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/><title>ClinicalChat — Sign up</title>
<link rel="icon" href="/brand/logo.svg"/>
<style>
*{{box-sizing:border-box}} body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6fb;color:#111827}}
.card{{width:min(560px,94vw);background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:28px 32px;
box-shadow:0 10px 30px rgba(17,24,39,.06)}}
.brand{{text-align:center;margin-bottom:14px}} .brand img{{width:56px;height:56px}}
h1{{font-size:22px;margin:6px 0 2px}} .sub{{font-size:13px;color:#6b7280}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px 14px;margin-top:16px}} .full{{grid-column:1/-1}}
label{{display:block;font-size:12.5px;font-weight:600;color:#374151;margin-bottom:4px}}
input,select{{width:100%;padding:9px 11px;border:1px solid #d1d5db;border-radius:9px;font-size:14px;background:#fff}}
input:focus,select:focus{{outline:2px solid #c7d2fe;border-color:#6366f1}}
.hint{{font-size:11.5px;color:#9ca3af;margin-top:3px}}
button{{width:100%;margin-top:18px;padding:11px;border:0;border-radius:10px;background:#4f46e5;color:#fff;font-size:15px;
font-weight:600;cursor:pointer}} button:hover{{background:#4338ca}}
.err{{background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;border-radius:9px;padding:9px 12px;font-size:13.5px;margin-top:12px}}
.ok{{background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;border-radius:9px;padding:12px 14px;font-size:14px;margin-top:12px}}
.foot{{text-align:center;font-size:13px;color:#6b7280;margin-top:14px}} a{{color:#4f46e5;font-weight:600;text-decoration:none}}
.note{{font-size:12px;color:#6b7280;background:#f9fafb;border:1px solid #eef0f4;border-radius:9px;padding:8px 11px;margin-top:14px}}
</style></head><body><div class="card">
<div class="brand"><img src="/brand/logo.svg" alt=""/><h1>Create your patient account</h1>
<div class="sub">Tell us a little about yourself — then describe your symptoms and upload any reports in the chat.</div></div>
{body}
<div class="foot">Already have an account? <a href="/">Log in</a></div></div></body></html>"""

FORM = """{error}<form method="post" action="/signup" autocomplete="off"><div class="grid">
<div class="full"><label for="full_name">Full name</label><input id="full_name" name="full_name" required maxlength="80" value="{full_name}"/></div>
<div><label for="age">Age</label><input id="age" name="age" type="number" min="1" max="119" required value="{age}"/></div>
<div><label for="gender">Gender</label><select id="gender" name="gender">{gender_options}</select></div>
<div class="full"><label for="username">Username</label><input id="username" name="username" required maxlength="32" value="{username}"/>
<div class="hint">lowercase letters, digits, '.', '_' or '-'</div></div>
<div><label for="password">Password</label><input id="password" name="password" type="password" required minlength="8"/>
<div class="hint">at least 8 characters</div></div>
<div><label for="confirm">Confirm password</label><input id="confirm" name="confirm" type="password" required minlength="8"/></div>
<div class="full"><label for="allergies">Known allergies (optional)</label><input id="allergies" name="allergies" placeholder="e.g. penicillin, peanuts" value="{allergies}"/></div>
<div class="full"><label for="medications">Current medicines (optional)</label><input id="medications" name="medications" placeholder="e.g. metformin, amlodipine" value="{medications}"/></div>
</div><button type="submit">Create account</button>
<div class="note">🔒 Only you and your care team can see your record. ClinicalChat gives information to support — not replace —
your doctor. In an emergency call your local emergency number.</div></form>"""

DONE = """<div class="ok">✅ <b>Account created.</b><br/>Your username is <b>{username}</b>. Log in with the password you chose,
then tell the assistant how you feel or upload a report with the 📎 button.</div>
<a href="/"><button type="button">Go to login</button></a>"""


def render_form(values: Optional[Dict[str, str]] = None, error: str = "") -> str:
    v = {k: html.escape(str(x or "")) for k, x in (values or {}).items()}
    gender = (values or {}).get("gender", "")
    options = "".join(f"<option value='{g}'{' selected' if g == gender else ''}>{g or 'Prefer not to say'}</option>"
                      for g in ("", "Female", "Male", "Other"))
    body = FORM.format(error=f"<div class='err'>{html.escape(error)}</div>" if error else "", gender_options=options,
                       **{k: v.get(k, "") for k in ("full_name", "age", "username", "allergies", "medications")})
    return PAGE.format(body=body)


def render_done(username: str) -> str:
    return PAGE.format(body=DONE.format(username=html.escape(username)))
