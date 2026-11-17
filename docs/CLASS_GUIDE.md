# ClinicalChat — Class Guide

> **Generated**: 2026-11-16 17:30 UTC+5:30
> **Updated**: 2026-11-16 17:30 UTC+5:30
> **Status**: Final
> **Audience**: anyone who wants to read the code for the first time — where it starts, what every class does, who uses it.

## Table of Contents
- [1. Where does it start?](#1-where-does-it-start)
- [2. The startup chain, step by step](#2-the-startup-chain-step-by-step)
- [3. Where are the agents configured?](#3-where-are-the-agents-configured)
- [4. How is an agent called? (life of one question)](#4-how-is-an-agent-called-life-of-one-question)
- [5. Every class: responsibility and where it is used](#5-every-class-responsibility-and-where-it-is-used)
  - [5.1 Start-up and wiring](#51-start-up-and-wiring)
  - [5.2 Web layer (what you click)](#52-web-layer-what-you-click)
  - [5.3 Services (the workflow)](#53-services-the-workflow)
  - [5.4 Agents](#54-agents)
  - [5.5 Domain (the shared vocabulary)](#55-domain-the-shared-vocabulary)
  - [5.6 Knowledge and ingestion](#56-knowledge-and-ingestion)
  - [5.7 Infrastructure (talks to the outside world)](#57-infrastructure-talks-to-the-outside-world)
- [6. Reading order for a newcomer](#6-reading-order-for-a-newcomer)
- [Revision History](#revision-history)

Related: [CODE_FLOW.md](CODE_FLOW.md) (line-by-line code flow) · [ARCHITECTURE.md](ARCHITECTURE.md) (diagrams) · [FILE_GUIDE.md](FILE_GUIDE.md) (every file) ·
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) (plain-English tour) · [COURSE_ALIGNMENT.md](COURSE_ALIGNMENT.md) (Week 1–8 mapping).

---

## 1. Where does it start?

There is no single "main class". The program starts at a **function**, and the first **class** it builds is the
`Container`.

| What | Where | Why it matters |
|------|-------|----------------|
| The command you type | `clinicalchat serve` | Declared in `pyproject.toml` → `[project.scripts] clinicalchat = "clinicalchat.cli:main"` |
| Entry function | [`cli.py`](../src/clinicalchat/cli.py) → `main()` → `_serve()` | Reads the sub-command (`serve`, `ingest`, `ask`, `reset-patient`) |
| Web start | [`web/app.py`](../src/clinicalchat/web/app.py) → `serve()` | Builds everything once, then starts the web server (uvicorn) |
| **Starting class** | [`container.py`](../src/clinicalchat/container.py) → `build_container()` returns a **`Container`** | The *composition root*: the ONE place that creates every agent and service and plugs them together |
| Web app | `web/app.py` → `create_app(container)` | A FastAPI app with the Gradio chat page mounted on `/` and the REST API on `/api/*` |

Think of `Container` as the **switchboard**: every other class receives what it needs from the container through its
constructor (dependency injection). That is why tests can replace OpenAI with a `FakeLLM` by passing
`build_container(llm=fake)`.

---

## 2. The startup chain, step by step

```
clinicalchat serve
 └─ cli.main()                       parse "serve"
     └─ cli._serve()
         └─ web.app.serve()
             ├─ container.get_container()          build once, reuse forever
             │    └─ build_container(settings)
             │         ├─ Settings  (config.py)    read .env (OpenAI key, models, Langfuse, Tavily…)
             │         ├─ Database + Repositories  open data/clinicalchat.db (SQLite)
             │         ├─ Tracer                   Langfuse (or a silent no-op)
             │         ├─ KnowledgeBase.load()     parse dataset → catalog, patients, knowledge graph, vector index
             │         │    └─ load_user_data()    re-add uploads, sign-ups, patient facts from the database
             │         ├─ create_chat_model() ×3   OpenAI: insight (gpt-4.1), fast (gpt-4.1-mini), specialist (gpt-4.1-mini)
             │         ├─ the 5 agents + SpecialistPanel
             │         └─ the services (Router, AnalysisPipeline, responders, ChatService, UploadService…)
             ├─ warm_up()                         one tiny LLM call in the background (first answer is faster)
             └─ uvicorn.run(create_app(container))
                  ├─ FastAPI + lifespan            on stop: finish background work, flush Langfuse
                  ├─ /api/* REST routes            web/api.py
                  └─ Gradio UI on "/"              web/layout.build_ui(UIController(container))
```

---

## 3. Where are the agents configured?

| Question | Answer | File |
|----------|--------|------|
| Which agents exist? | `AgentName` enum: `EXTRACTION`, `ANALYSIS`, `RISK`, `INSIGHT`, `COMMUNICATION` + the AI specialists `LAB_INTERPRETER`, `RADIOLOGY_ANALYZER`, `ALLERGY_SAFETY` | [`domain/roles.py`](../src/clinicalchat/domain/roles.py) |
| Which agents run for which role, and which records each role may see? | `POLICIES` — one `RolePolicy` per role (agents, record domains, own-record-only, can-upload, writing style) | `domain/roles.py` |
| Where is each agent object created and given its model? | `build_container()` — the `# agents` block | [`container.py`](../src/clinicalchat/container.py) |
| Which OpenAI model does each agent use? | `create_chat_model(settings, INSIGHT / FAST / SPECIALIST)`; model names come from `.env` (`OPENAI_MODEL`, `OPENAI_FAST_MODEL`, `OPENAI_SPECIALIST_MODEL`) | [`infrastructure/llm/models.py`](../src/clinicalchat/infrastructure/llm/models.py), [`config.py`](../src/clinicalchat/config.py) |
| What does each AI agent "say" to the model? | The prompts: `INSIGHT_SYSTEM` / `INSIGHT_PROMPT`, `COMMUNICATION_SYSTEM` / `COMMUNICATION_PROMPT`, `GENERAL_SYSTEM`, `GROUND_RULES`; the specialists' `LAB_SYSTEM`, `RADIOLOGY_SYSTEM`, `ALLERGY_SYSTEM`, `SPECIALIST_PROMPT` sit next to their classes | [`agents/llm/prompts.py`](../src/clinicalchat/agents/llm/prompts.py), [`agents/specialists/ai.py`](../src/clinicalchat/agents/specialists/ai.py) |
| What shape must the answer have? | A Pydantic `output_model` per agent (`InsightOutput`, `CommunicationOutput`, `LabPanel`, …) | each agent file |
| Turn the AI specialists on / off | `SPECIALIST_AGENTS=auto \| llm \| rules` in `.env` → `Settings.specialists_use_llm` → `SpecialistPanel(enabled=…)` | `config.py`, `container.py` |

The agents block in `build_container()`:

```python
extraction = DataExtractionAgent(kb.catalog, kb.patients, kb.kg, kb.records)      # rules, no AI
analysis = ClinicalAnalysisAgent(kb.catalog, kb.patients, kb.kg, kb.safety)       # rules, no AI
insight_agent = InsightGenerationAgent(insight_llm)                               # AI (gpt-4.1)
communication_agent = PatientCommunicationAgent(fast_llm)                         # AI (gpt-4.1-mini)
specialists = SpecialistPanel(specialist_llm, enabled=settings.specialists_use_llm)  # 3 AI specialists
pipeline = AnalysisPipeline(extraction, analysis, RiskDetectionAgent(), ...)      # Risk = rules, no AI
```

**Two kinds of agents:**

| Kind | Agents | How they work |
|------|--------|---------------|
| Rule-based (deterministic, auditable) | 📄 Data Extraction, 🔬 Clinical Analysis, 🚨 Risk Detection | Plain Python: reference ranges, knowledge-graph lookups, safety rules. Same input → same output. No cost, milliseconds. |
| AI (LLM) | 💡 Insight Generation, 💬 Patient Communication, 🧪 Lab Interpreter, 🩻 Radiology Analyzer, 🛡️ Allergy Safety | Subclasses of `LLMAgent`: one LangChain call `prompt \| model`, answer validated against a Pydantic model, timed and traced. |

---

## 4. How is an agent called? (life of one question)

Example: Dr. Sharma asks *"what medicine does Sowmya need to control the thyroid?"*

| # | Who | What happens | File |
|---|-----|--------------|------|
| 1 | Browser → `UIController.respond()` | The Gradio "send" event calls the controller | `web/controller.py` |
| 2 | `ChatService.ask(question, session)` | The single entry point for every chat message; returns a *stream* of `Event`s | `services/chat.py` |
| 3 | `TurnRecorder.start()` | Opens a Langfuse trace for this question | `services/turn_recorder.py` |
| 4 | `Router.route()` | Keywords, no AI: whose records? which record types (lab)? allowed for this role? which mode (case / general / explain / status)? → `RoutePlan` | `services/router.py` |
| 5 | `ChatService._case()` | Staff path: `pipeline.run(...)` then `StaffResponder.respond(...)` | `services/chat.py` |
| 6 | `AnalysisPipeline._extract()` → **📄 `DataExtractionAgent.run()`** | Picks the patient's relevant reports (the two TSH reports), scores how well they were read | `services/pipeline.py`, `agents/extraction.py` |
| 7 | `AnalysisPipeline._analyse()` → **🔬 `ClinicalAnalysisAgent.run()`** | Values vs reference ranges (TSH 11.2 HIGH), imaging red flags, allergy/interaction checks via the knowledge graph | `agents/analysis.py` |
| 7a | `StaffResponder.prefetch()` | Starts the guideline web search *now*, in parallel | `services/responders.py` |
| 7b | **`SpecialistPanel.run()`** → 🧪 `LabInterpreterAgent` (+ others the role may see) | AI second opinion, in parallel threads | `services/specialist_panel.py`, `agents/specialists/ai.py` |
| 8 | `AnalysisPipeline._score()` → **🚨 `RiskDetectionAgent.run()`** | Safety rules → LOW / MODERATE / HIGH / CRITICAL | `agents/risk.py` |
| 9 | `StaffResponder.respond()` → `InsightService.write()` → **💡 `InsightGenerationAgent.run()`** | One AI call with the findings + record excerpts + web evidence → structured answer | `services/responders.py`, `services/insight_service.py`, `agents/insight.py` |
| 10 | `LLMAgent._invoke()` | Builds inputs, calls `prompt \| llm`, retries once if the answer was cut off, validates JSON, never crashes the chat | `agents/llm/base.py` |
| 11 | `chat_renderer.render()` | Turns each `Event` into the live activity log and the final answer bubble | `web/chat_renderer.py` |
| 12 | `TurnRecorder.finish()` | Saves history, writes the audit line, scores the Langfuse trace | `services/turn_recorder.py` |

**Patient path differences:** after step 8, `PatientResponder` answers through **💬 `PatientCommunicationAgent`** (plain
language, checked by the guard: no doses, no definitive diagnosis), files a case for the care team (`CaseFiler`) and
runs the Insight + AI specialists **in the background** (`InsightService.in_background`) so the patient does not wait.

**REST API path:** `POST /api/query` → `ChatService.run()` (the same `ask()` collected into a list) — same agents, same
access rules.

---

## 5. Every class: responsibility and where it is used

"Built by" = who creates the object. Almost everything is built once in `build_container()`.

### 5.1 Start-up and wiring

| Class | File | Responsibility | Built by / used by |
|-------|------|----------------|--------------------|
| `Settings` | `config.py` | All configuration from `.env`: API keys, model names, folders, limits, feature switches | `get_settings()`; read by the container, LLM factory, tracer, vector index, web search |
| `Container` | `container.py` | Holds every ready-made service (chat, uploads, care team, insight, accounts, admin…) | `build_container()`; used by `web/app.py`, `web/api.py`, `UIController`, `Sidebar`, `cli.py`, tests |
| `Repositories` | `container.py` | Bundles the database repositories (users, facts, uploads, cases, feedback, caches) | `Repositories.on(Database)` inside `build_container()`; `cli.py` (reset-patient) |

### 5.2 Web layer (what you click)

| Class | File | Responsibility | Built by / used by |
|-------|------|----------------|--------------------|
| `UIController` | `web/controller.py` | Gradio event handlers: each click calls ONE container service and returns screen updates (send, upload, pick patient, approve case, allergy check, poll) | `create_app()`; wired to buttons in `web/layout.build_ui()` |
| `Sidebar` | `web/sidebar.py` | Builds the sidebar HTML per role: patient card, my cases, work queue, case board | `UIController` |
| `ActivityLog` | `web/chat_renderer.py` | The collapsible "agent activity" message updated live while agents work | `_Turn` |
| `_Turn` | `web/chat_renderer.py` | Chat state of one question while its events arrive (status → steps → partial → answer) | `render()` (called by `UIController.respond`) |
| `QueryRequest` / `QueryResponse` | `web/api.py` | JSON body and reply of `POST /api/query` | FastAPI (Swagger at `/docs`) |
| `AllergyCard` | `web/api.py` | Reply of `GET /api/allergy/{allergen}` | FastAPI |
| `ApiSessions` | `web/api.py` | Remembers API conversations per user + session id so follow-up questions work | `api_router(container)`, included by `create_app()` |

### 5.3 Services (the workflow)

| Class | File | Responsibility | Built by / used by |
|-------|------|----------------|--------------------|
| `ChatService` | `services/chat.py` | **Answers one chat message.** Routes it, then hands it to the right mode handler (status / general / explain / case) | Container; `UIController.respond`, `/api/query`, `cli ask` |
| `SymptomRecorder` | `services/chat.py` | Saves symptoms a patient mentions to the database + knowledge graph | `ChatService`, `UploadService` |
| `Router` | `services/router.py` | Keyword router (no AI): patient, record types, RBAC, mode, drugs/allergens/symptoms mentioned | `ChatService`, `UploadService` |
| `RoutePlan` | `services/router.py` | The router's decision for one message | Passed through every later step |
| `AnalysisPipeline` | `services/pipeline.py` | Steps 1–3: **📄 Extraction → 🔬 Analysis (+ AI specialists for staff) → 🚨 Risk**, each traced | `ChatService._case`, `UploadService` |
| `SpecialistPanel` | `services/specialist_panel.py` | Runs the 3 AI specialists in parallel, only for record types the role may see | `AnalysisPipeline` (staff), `InsightService` (patient cases, background) |
| `StaffResponder` | `services/responders.py` | Staff answer: guideline web search (prefetched) + **💡 Insight** | `ChatService`, `UploadService` |
| `PatientResponder` | `services/responders.py` | Patient answer: reuse cached Insight or rule-based findings → **💬 Communication**; files the case; background Insight | `ChatService`, `UploadService` |
| `GeneralResponder` | `services/responders.py` | General health questions (no personal records): FAQ cache, trusted web sources, Communication agent | `ChatService._general` |
| `InsightService` | `services/insight_service.py` | Runs the Insight agent (staff: while they wait; patient cases: in the background), caches it, `drain()` on shutdown | `StaffResponder`, `PatientResponder`, `web/app.py` lifespan |
| `Communicator` | `services/communicator.py` | Writes the patient answer, runs the guard `agents/guard.py` (no doses / definitive diagnosis / medicine changes), adds urgent-help guidance | `PatientResponder`, `StaffResponder` (explain), `ChatService._explain` |
| `AnswerWriter` | `services/answer_writer.py` | Streams the Communication agent's answer word by word into the chat | `Communicator`, `GeneralResponder` |
| `CaseFiler` | `services/cases.py` | Opens/updates the patient's ONE care-team case and routes it (radiologist → coordinator → clinician) | `PatientResponder`, `Communicator` |
| `CareTeamService` | `services/care_team.py` | Care-team actions: radiology read, triage, clinician approve/edit, work queues, delivery to the patient | Container; `UIController.decide/open_case`, `Sidebar` |
| `UploadService` | `services/uploads.py` | Reads attached files in parallel (OCR for photos), checks the name, saves, then runs the same pipeline | Container; `UIController.respond_with_files` |
| `WebEnrichment` | `services/web_enrichment.py` | Tavily searches on trusted sites: unknown drug names, general answers, guideline evidence (can start early in parallel) | `ChatService`, `StaffResponder`, `GeneralResponder` |
| `KnowledgeBase` | `services/knowledge_base.py` | Everything the app knows, in memory: catalog, patients, knowledge graph, vector index, records, safety — kept in sync when uploads arrive | `build_container()`; `UploadService`, `PatientDataAdmin`, `cli ingest` |
| `DocumentCatalog` | `services/catalog.py` | Index of every document and section: its type (lab/radiology/…), patient and file name | `KnowledgeBase`; Extraction & Analysis agents, `CaseFiler` |
| `PatientDirectory` | `services/patients.py` | Patient profiles (dataset + sign-ups), name lookup in text, the summary given to the AI | `KnowledgeBase`; router, agents, responders, accounts |
| `RecordRetriever` | `services/records.py` | Builds the compact record text the AI reads (selected sections + vector search, max 9000 chars) | Extraction agent, `InsightService`, `GeneralResponder` |
| `SafetyChecker` | `services/safety.py` | Allergy, cross-reactivity, interaction and contrast-dye checks (knowledge graph, no AI) | Analysis agent, pipeline safety banner, `CaseFiler`, API allergy card |
| `AccountService` | `services/accounts.py` | Logins: demo personas + self-registered patients (hashed passwords), sign-up | Container; Gradio login, `/api/*` basic auth, `/signup` |
| `PatientDataAdmin` | `services/admin.py` | Wipes one patient's uploads, facts, cases, cached insight and exports | Container; `clinicalchat reset-patient` |
| `UploadIndex` | `services/admin.py` | Interface (Protocol): "anything that can forget a patient's uploads" | Type of `PatientDataAdmin`'s index (VectorIndex or KnowledgeBase) |
| `SessionExporter` | `services/export.py` | Exports a chat as a Markdown summary (the ⬇️ Export button) | Container; `UIController.after_turn`, admin |
| `TurnRecorder` | `services/turn_recorder.py` | Opens the Langfuse trace for a turn and, at the end, saves history, audit line and scores | `ChatService`, `UploadService` |
| `TurnContext` | `services/turn.py` | What one turn carries: question, session, plan, trace, and the `PipelineRun` | Created in `ChatService.ask`; passed to every step |
| `PipelineRun` | `services/turn.py` | Everything a run produced: extraction, analysis, risk, AI results, case flags, web prefetch | Inside `TurnContext` |

### 5.4 Agents

| Class | File | Responsibility | Built by / used by |
|-------|------|----------------|--------------------|
| `DataExtractionAgent` | `agents/extraction.py` | **📄 Agent 1 (rules).** Choose the reports relevant to the question within the role's access; quality score; list missing data | Container → `AnalysisPipeline._extract` |
| `ExtractionResult` | `agents/extraction.py` | Its output: sections, sources, symptoms, missing data, quality | Analysis, Risk, Insight, presenters |
| `ClinicalAnalysisAgent` | `agents/analysis.py` | **🔬 Agent 2 (rules).** Combines the lab / radiology / allergy / symptom specialists into findings | Container → `AnalysisPipeline._analyse` |
| `AnalysisResult` | `agents/analysis.py` | Its output: abnormal labs, imaging findings, alerts, possible conditions, AI specialist report | Risk, Insight, presenters, `SpecialistPanel` |
| `LabFindings` | `agents/specialists/lab.py` | Rule-based lab specialist result: abnormal values and patterns | `ClinicalAnalysisAgent` |
| `ImagingFinding` | `agents/specialists/radiology.py` | One red flag / finding found in an imaging report (negation-aware) | `ClinicalAnalysisAgent`, Risk |
| `AllergySafety` / `AllergyFindings` | `agents/specialists/allergy.py` | Rule-based allergy check of drugs asked about and current medicines, and its result | `ClinicalAnalysisAgent` |
| `SymptomReasoner` | `agents/specialists/symptoms.py` | Which conditions could explain the symptoms; "clinical inconsistency" checks | `ClinicalAnalysisAgent` |
| `RiskDetectionAgent` | `agents/risk.py` | **🚨 Agent 3 (rules).** Safety rules → risk level, indicators, escalate yes/no | Container → `AnalysisPipeline._score` |
| `LLMAgent` | `agents/llm/base.py` | **Base class of every AI agent**: `prompt \| llm`, streaming, retry on cut-off, JSON validation, timing, tracing, never crashes | Parent of Insight, Communication, Specialist agents |
| `AgentResult` | `agents/llm/base.py` | What any AI agent returns: output dict, parsed ok?, error, latency, cache hit | Responders, presenters, Langfuse scores |
| `InsightGenerationAgent` | `agents/insight.py` | **💡 Agent 4 (AI, gpt-4.1).** Consolidated clinical answer: key findings, possible conditions, next steps, confidence | `InsightService` |
| `InsightOutput` | `agents/insight.py` | The answer shape the model must return | `InsightGenerationAgent.output_model` |
| `PatientCommunicationAgent` | `agents/communication.py` | **💬 Agent 5 (AI, gpt-4.1-mini, streamed).** Plain-language answer for patients | `AnswerWriter` |
| `CommunicationOutput` | `agents/communication.py` | Its answer shape | `PatientCommunicationAgent.output_model` |
| `SpecialistAgent` | `agents/specialists/ai.py` | Base of the course AI specialists: report text in → list of course Pydantic objects out | Parent of the three below |
| `LabInterpreterAgent` | `agents/specialists/ai.py` | 🧪 AI lab interpretation (course Week 4) | `SpecialistPanel` |
| `RadiologyAnalyzerAgent` | `agents/specialists/ai.py` | 🩻 AI radiology interpretation | `SpecialistPanel` |
| `AllergySafetyAgent` | `agents/specialists/ai.py` | 🛡️ AI allergy safety assessment | `SpecialistPanel` |
| `LabInterpretation`, `RadiologyFinding`, `AllergyAssessment` | `agents/specialists/ai.py` | The course notebook's exact output models | Inside the panels below |
| `LabPanel`, `RadiologyReview`, `AllergyReview` | `agents/specialists/ai.py` | "A list of" the above — what each specialist returns | `SpecialistPanel` |
| `SpecialistReport` | `agents/specialists/ai.py` | What the specialists said this turn + urgent items + timings | `AnalysisResult.ai`, presenters, case views |

### 5.5 Domain (the shared vocabulary)

| Class | File | Responsibility | Used by |
|-------|------|----------------|---------|
| `Role` | `domain/roles.py` | patient / clinician / radiologist / coordinator | Everywhere access is decided |
| `AgentName` | `domain/roles.py` | Names of all agents (also the labels with emoji) | Agents, pipeline, UI |
| `RolePolicy` | `domain/roles.py` | Per role: allowed agents, record types, own-record-only, can upload, writing style | Router, Extraction, uploads, prompts |
| `Persona` | `domain/personas.py` | A built-in demo login (name, role, patient) | `AccountService`, UI, API, CLI |
| `ChatSession` | `domain/conversation.py` | A logged-in conversation: who, which patient, history, last insight | Every service call |
| `Turn` | `domain/conversation.py` | One question and its answers (for history, audit, export) | `TurnContext`, exporter |
| `Event` | `domain/conversation.py` | One live update streamed to the screen (status, step, safety, partial, answer, blocked) | Produced by services, consumed by `chat_renderer` / API |
| `Domain` | `domain/models.py` | Record types: lab, radiology, allergy, patient_record, reference | Catalog, RBAC, router, extraction |
| `LabStatus` | `domain/models.py` | LOW / NORMAL / HIGH / CRITICAL_LOW / CRITICAL_HIGH / UNKNOWN | Lab parsing, risk |
| `LabResult` | `domain/models.py` | One lab value with unit, reference range and computed status | Ingestion, analysis, specialists |
| `Section` | `domain/models.py` | One readable piece of a report, belonging to one patient | Everything that reads records |
| `ClinicalDocument` | `domain/models.py` | A whole report (dataset or upload) made of sections | Ingestion, catalog, vector index |
| `AllergyRecord` | `domain/models.py` | One documented allergy (severity, reactions, alternatives) | Registry, patient directory |
| `PatientProfile` | `domain/models.py` | Everything known about a patient: demographics, conditions, medicines, allergies | Agents, responders, UI |
| `SafetyAlert` | `domain/models.py` | An allergy / interaction / contrast warning with severity | Safety checks, risk, answers |
| `RiskIndicator` / `RiskAssessment` | `domain/risk.py` | One risk signal / the overall risk result | Risk agent, presenters, cases |
| `WebResult` / `WebSearchOutcome` | `domain/web_evidence.py` | One trusted-site result / a search's results, shown as [W1] to the AI and as links to people | `TavilySearch`, `WebEnrichment`, Insight |

### 5.6 Knowledge and ingestion

| Class | File | Responsibility | Used by |
|-------|------|----------------|---------|
| `MedicalKnowledgeGraph` | `knowledge/graph.py` | NetworkX graph of drugs, allergens, conditions, symptoms and each patient's facts; drug resolution, cross-reactivity, interactions | `KnowledgeBase`; router, agents, safety |
| `PatientRegistry` | `ingestion/registry.py` | Works out which patient each document section belongs to and extracts safety-critical facts | `load_corpus`, `KnowledgeBase`, `PatientDirectory` |
| `Block` | `ingestion/loaders.py` | One piece of raw text/table read from DOCX / PDF / XLSX / image (OCR) | `sectioning.py` |
| `_Draft` / `SectionDraft` | `ingestion/sectioning.py` | Work-in-progress and finished section while splitting a document per patient | `split_sections()` (uploads + dataset) |
| `SectionQuality` / `ExtractionQuality` | `ingestion/quality.py` | How far we trust what was read (score 0–1, issues) | Extraction agent, uploads |

### 5.7 Infrastructure (talks to the outside world)

| Class | File | Responsibility | Used by |
|-------|------|----------------|---------|
| `Database` | `infrastructure/database.py` | One SQLite file, schema, thread-safe writes | All repositories |
| `UserRepository` | `infrastructure/repositories/users.py` | Self-registered patient logins (PBKDF2 hashes) | `AccountService`, admin |
| `PatientFactRepository` | `…/patient_facts.py` | Symptoms, allergies, medicines patients told us | Accounts, `SymptomRecorder`, cases, admin |
| `UploadRepository` | `…/uploads.py` | Uploaded reports (parsed JSON) + quality score | `UploadService`, cases, admin, start-up reload |
| `CaseRepository` | `…/cases.py` | Care-team cases and their payload | `CaseFiler`, `CareTeamService`, `InsightService`, admin |
| `FeedbackRepository` | `…/feedback.py` | Clinician review notes, fed into later Insight prompts | `CareTeamService`, `InsightService`, admin |
| `InsightCache` | `…/caches.py` | Latest Insight per patient (reused until new evidence) | `InsightService`, admin |
| `FaqCache` | `…/caches.py` | Answers to general questions | `GeneralResponder` |
| `WebCache` | `…/caches.py` | Web search results kept for a week | `TavilySearch` |
| `SQLiteLLMCache` | `infrastructure/llm/cache.py` | Identical prompt → saved answer (no second OpenAI charge) | `install_cache()` in `build_container()` |
| `UnavailableChatModel` | `infrastructure/llm/models.py` | Stand-in model when no OpenAI key is set: fails politely | `create_chat_model()` |
| `Tracer` / `Span` | `infrastructure/tracing.py` | Langfuse traces, steps and scores (silent no-op without keys) | Container; every service step |
| `VectorIndex` / `SearchHit` | `infrastructure/vector_index.py` | ChromaDB: find passages similar in meaning (OpenAI embeddings, local fallback) | `KnowledgeBase`, `RecordRetriever`, `cli ingest`, admin |
| `TavilySearch` | `infrastructure/web_search.py` | Tavily web search limited to trusted medical sites | `WebEnrichment` |
| `AuditLog` | `infrastructure/audit_log.py` | One JSON line per question / upload / care-team action | `TurnRecorder`, `CareTeamService` |

---

## 6. Reading order for a newcomer

1. `cli.py` → `web/app.py` (`serve`, `create_app`) — how it starts.
2. `container.py` — every object and how they connect.
3. `domain/roles.py` — roles, agents, access rules.
4. `services/chat.py` (`ChatService.ask`) → `services/router.py` → `services/pipeline.py` → `services/responders.py` — one question end to end.
5. `agents/llm/base.py` then `agents/insight.py` — how an AI agent call works.
6. `agents/extraction.py`, `agents/analysis.py`, `agents/risk.py` — the rule-based agents.
7. `services/cases.py` + `services/care_team.py` — the radiologist → coordinator → clinician workflow.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2026-11-16 | Copilot | Initial version: entry point, start-up chain, agent configuration, life of a question, all classes |
| v1.1 | 2026-11-16 | Copilot | Link to the new CODE_FLOW.md walkthrough |
