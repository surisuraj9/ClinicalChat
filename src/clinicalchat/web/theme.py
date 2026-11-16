"""Look and feel: the page CSS (full-height layout, light palette, agent strip, case cards) and a light-theme switch."""

LOGO_URL = "/brand/logo.svg"

CSS = """
/* ---------- full-width, full-height layout ---------- */
.gradio-container {max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 12px 16px 4px !important;
  background: #f4f6fa !important;}
.gradio-container > .main, .gradio-container .wrap {max-width: 100% !important;}
.gradio-container > .main {padding: 0 !important;}
.gradio-container .contain {gap:12px !important;}
.cc-top {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:8px 16px !important; align-items:center !important;
  gap:12px; min-height:64px; flex-wrap:nowrap !important;}
.cc-brand {display:flex; align-items:center; gap:10px;}
.cc-brand img {width:38px; height:38px;}
.cc-brand .t {font-weight:700; font-size:19px; color:#111827; line-height:1.1;}
.cc-brand .s {font-size:12.5px; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-brand > div {min-width:0;}
.cc-user {display:flex; align-items:center; justify-content:flex-end; gap:10px;}
.cc-avatar {width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#0d9488); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; flex:none;}
.cc-user .n {font-weight:600; color:#111827; font-size:14px; line-height:1.15;}
.cc-user .r {font-size:12px; color:#6b7280;}
.cc-pill {display:inline-block; padding:1px 8px; border-radius:999px; background:#eef2ff; color:#4338ca; font-size:11.5px; font-weight:600;}
.cc-main {gap:16px !important; flex-wrap:nowrap !important;}

/* ---------- sidebar: identical full-width tiles ---------- */
.cc-side {flex: 0 0 340px !important; width:340px !important; max-width:340px !important; height: calc(100vh - 116px);
  overflow-y:auto; overflow-x:hidden; gap:12px !important; padding-right:2px;
  flex-direction:column !important; flex-wrap:nowrap !important;}
.cc-chatcol {flex: 1 1 auto !important;}
.cc-side > .cc-tile:last-child {flex:1 1 auto !important;}
.cc-tile {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:14px !important;
  padding:12px 14px !important; gap:6px !important; width:100% !important; box-shadow:0 1px 2px rgba(16,24,40,.04); flex:none !important;}
.cc-tile .cc-h {font-size:12px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:.06em; margin:0;}
.cc-tile .block, .cc-tile .form {border:none !important; box-shadow:none !important; padding:0 !important; background:transparent !important;}
.cc-pname {font-weight:600; font-size:15px; color:#111827;}
.cc-pmeta {font-size:12.5px; color:#6b7280; margin:2px 0 6px;}
.cc-allergy, .cc-ok {border-radius:10px; padding:7px 10px; font-size:13px; margin-top:8px;}
.cc-allergy {background:#fef2f2; border:1px solid #fecaca; color:#991b1b;}
.cc-ok {background:#f0fdf4; border:1px solid #bbf7d0; color:#166534;}
.cc-tag {display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:999px; background:#f1f5f9; color:#334155; font-size:12px;}
.cc-muted {font-size:12.5px; color:#6b7280;}
button.cc-chip {justify-content:flex-start !important; text-align:left !important; font-weight:500 !important; width:100% !important;
  min-height:42px !important; height:42px !important; padding:6px 12px !important; line-height:1.3 !important;
  background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; font-size:13px !important;
  white-space:normal !important; border-radius:10px !important; box-shadow:none !important;}
button.cc-chip:hover {background:#eef2ff !important; border-color:#c7d2fe !important;}

/* ---------- agent strip: which agent is working on the question ---------- */
.cc-agents {display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:8px;}
.cc-agent {background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:8px 10px; height:62px; display:flex;
  flex-direction:column; justify-content:center; transition:all .2s; overflow:hidden;}
.cc-agent .an {font-weight:600; font-size:13px; color:#111827; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-agent .as {font-size:12px; color:#6b7280; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-agent.locked {background:#f8fafc; opacity:.6;}
.cc-agent.queued {border-color:#c7d2fe; background:#f5f7ff;}
.cc-agent.working {border-color:#6366f1; background:#eef2ff; box-shadow:0 0 0 3px rgba(99,102,241,.15); animation:ccpulse 1.4s infinite;}
.cc-agent.working .as {color:#4338ca; font-weight:600;}
.cc-agent.done {border-color:#86efac; background:#f0fdf4;}
.cc-agent.done .as {color:#166534; font-weight:600;}
.cc-agent.failed {border-color:#fca5a5; background:#fef2f2;}
.cc-agent.failed .as {color:#991b1b;}
.cc-agent.denied {border-color:#fcd34d; background:#fffbeb; opacity:1;}
.cc-agent.skipped {background:#f8fafc; opacity:.75;}
.cc-agent .ak {font-size:10.5px; color:#9ca3af; font-weight:600; letter-spacing:.04em;}
.cc-case {font-size:12.5px; padding:6px 0; border-bottom:1px dashed #eef0f4;}
.cc-case .top {display:flex; justify-content:space-between; gap:8px;}
.cc-case .sub {color:#6b7280; font-size:11.5px; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cc-case:last-child {border-bottom:none;}
.cc-steps {display:flex; flex-wrap:wrap; gap:4px; margin-top:5px;}
.cc-steps span {font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid #e5e7eb; color:#9ca3af; background:#fff;}
.cc-steps span.done {color:#166534; border-color:#bbf7d0; background:#f0fdf4;}
.cc-steps span.now {color:#fff; border-color:#4f46e5; background:#4f46e5; font-weight:600;}
.cc-steps span.skip {text-decoration:line-through;}
.cc-board {display:grid; grid-template-columns:repeat(2, 1fr); gap:6px; margin-top:6px;}
.cc-bcell {border:1px solid #e5e7eb; border-radius:10px; padding:6px 8px; background:#f9fafb; display:flex; flex-direction:column;}
.cc-bcell b {font-size:18px; color:#111827; line-height:1.1;}
.cc-bcell span {font-size:11px; color:#6b7280;}
.cc-qrow {gap:8px !important; flex-wrap:nowrap !important;}
.cc-case .st {white-space:nowrap; font-weight:600;}
.cc-badge {display:inline-block; min-width:20px; padding:0 7px; border-radius:999px; background:#4f46e5; color:#fff; font-size:11px;
  font-weight:700; text-align:center; margin-left:6px; text-transform:none; letter-spacing:0;}
.cc-rev-btns {gap:8px !important; flex-wrap:nowrap !important;}
.cc-rev-btns button {min-width:0 !important; flex:1 1 0 !important; width:auto !important;
  height:36px !important; font-size:13px !important; font-weight:600 !important; border-radius:10px !important;}
.cc-tile textarea, .cc-tile input {font-size:13px !important;}
@keyframes ccpulse {0%,100% {box-shadow:0 0 0 3px rgba(99,102,241,.12);} 50% {box-shadow:0 0 0 6px rgba(99,102,241,.22);}}

/* ---------- chat area ---------- */
.cc-chatcol {gap:10px !important; min-width:0 !important;}
#cc-chat {border:1px solid #e5e7eb !important; border-radius:14px !important; background:#fff !important;}
.cc-input {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:6px 8px !important; align-items:center !important;
  flex-wrap:nowrap !important;}
.cc-input .block {border:none !important; box-shadow:none !important;}
.cc-input {gap:8px !important;}
.cc-input button, .cc-input .cc-up button {height:42px !important; white-space:nowrap !important; width:116px !important; min-width:116px !important;
  flex:0 0 116px !important; font-size:14px !important; font-weight:600 !important; padding:0 10px !important; border-radius:10px !important;}
.cc-input .cc-attach {flex:0 1 360px !important; max-width:40% !important; min-width:200px !important; overflow:visible !important;}
.cc-input .cc-attach .wrap {background:#eef2ff !important; border:1px solid #c7d2fe !important; border-radius:10px !important;
  padding:3px 4px !important; min-height:40px;}
.cc-input .cc-attach .wrap-inner {flex-wrap:nowrap !important; overflow-x:auto !important; gap:4px !important;
  scrollbar-width:thin; align-items:center;}
.cc-input .cc-attach .token {flex:0 0 auto !important; background:#fff !important; border:1px solid #c7d2fe !important;
  color:#3730a3 !important; font-size:12.5px !important; padding:2px 6px !important; margin:0 !important; white-space:nowrap;}
.cc-input .cc-attach .token-remove {color:#4338ca !important;}
.cc-input .cc-attach .secondary-wrap {flex:0 0 auto !important; width:auto !important;}
.cc-input .cc-attach .secondary-wrap input {display:none !important;}
.cc-input .cc-attach .icon-wrap {display:none !important;}
.cc-input button.cc-sec {background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#1f2937 !important; box-shadow:none !important;}
footer {display:none !important;}
@media (max-width: 900px) {.cc-main {flex-wrap:wrap !important;} .cc-side {height:auto;} .cc-agents {grid-template-columns:repeat(2, 1fr);}}
@media (max-width: 1280px) {.cc-input button {width:100px !important; min-width:100px !important; flex:0 0 100px !important;}}
"""

FORCE_LIGHT_JS = """
() => { const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }
"""
