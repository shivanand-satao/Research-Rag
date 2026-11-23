# Research RAG

A private research workspace with a neutral black interface, Supabase authentication, document-grounded chat, and persistent conversation history. The reference project is read-only; this application lives in the root `backend`, `frontend`, and `supabase` folders.

## Run locally

Python 3.12 and Node.js 22 are recommended. Local URLs belong in `frontend/.env` (`VITE_BACKEND_URL=http://127.0.0.1:8000`) and `backend/.env` (`FRONTEND_URL=http://127.0.0.1:5173`). The code also defaults to those localhost URLs. The browser obtains only Supabase public configuration from the configured backend's `/api/config` endpoint.

```powershell
# Terminal 1, from this directory
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2, from this directory
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open `http://127.0.0.1:5173`. On macOS/Linux, use `.venv/bin/python` and `npm` in the corresponding commands. If Vite uses another port, set `RESEARCH_REDIRECT_URL` and `CORS_ORIGINS` in `backend/.env`, and allow the matching callback URL in Supabase Auth settings.

`backend/requirements.txt` installs only the cloud runtime: the API, Qdrant client, MiniLM tokenizer, native PDF/Word extraction, and PDF preview support. It does not install Torch, transformers, sentence-transformers, OCR packages, or pytest. Pillow remains required for text/Word PDF previews. Cloud mode uses the `tokenizers` library and the MiniLM `tokenizer.json`; no local neural model weights are needed.

Optional installs are separate: `requirements-local.txt` for CPU cross-encoder reranking, `requirements-ocr.txt` for OCR (also needs the Tesseract executable), and `requirements-dev.txt` for tests. Reranking requires the optional local dependency group; OCR and reranking are disabled in the production profile.

## Vercel and Render

The checked-in production environment files contain public URLs and feature flags only; API keys remain in Render's Environment dashboard. Vite loads `frontend/.env.production` on a normal production build. The backend loads `backend/.env.production` when Render's `RENDER=true` or `APP_ENV=production` is set; otherwise it loads local `backend/.env`. Dashboard environment variables override these file defaults.

| Host | Setting | Value |
| --- | --- | --- |
| Vercel | Root Directory | `frontend` |
| Vercel | Build Command / Output | `npm run build` / `dist` |
| Vercel | `VITE_BACKEND_URL` | `https://research-rag-xmv4.onrender.com` |
| Render | Root Directory | `backend` |
| Render | Build Command | `pip install -r requirements.txt` |
| Render | Start Command | `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Render | Health Check Path | `/api/health` |
| Render | `APP_ENV` / `PYTHON_VERSION` | `production` / `3.12.12` |
| Render | `FRONTEND_URL` | `https://research-rag-steel.vercel.app` |
| Render | `RESEARCH_REDIRECT_URL` | `https://research-rag-steel.vercel.app/auth/callback` |
| Render | `CORS_ORIGINS` | `https://research-rag-steel.vercel.app` |
| Render | `RESEARCH_RERANK_ENABLED` / `OCR` | `false` / `false` |

`render.yaml` provides the backend deployment configuration. For an existing manually created service, copy the settings above or apply it as a Render Blueprint. Supply `SUPABASE_URL`, `SUPABASE_KEY` (or `SUPABASE_ANON_KEY`), `QDRANT_URL`, `QDRANT_API_KEY`, and `GROQ_API_KEY` in Render. Use a single backend worker because file-job coordination runs inside the process. Clear the Render build cache when switching from the old full dependency set, then redeploy. Redeploy Vercel after changing `VITE_BACKEND_URL`, since Vite embeds it at build time.

In Supabase Authentication URL Configuration, set the production Site URL to `https://research-rag-steel.vercel.app` and allow `https://research-rag-steel.vercel.app/auth/callback` plus the recovery variant ending in `?recovery=1`. Keep localhost entries if local development is still needed.

The frontend calls Render directly for authentication configuration, chat/file actions, and authenticated PDF previews. CORS allows the configured frontend origin. Vercel's configuration serves the SPA; it does not need an `/api` proxy. The backend root and `/api/health` return health JSON, and the root also accepts HEAD probes. A logged `404 GET /` in an older deployment means the root route was absent; it does not establish that the backend failed to start.

## Migrate Supabase

The migration is prepared in `supabase/migrations/20261109000000_research_rag.sql`; it has **not** been applied to the hosted database. It adds dedicated `research_documents`, `research_conversations`, and `research_messages` tables, ownership policies, an atomic chat-save function, and a private `research-documents` storage bucket.

Apply it through the Supabase SQL Editor, or use the Supabase CLI from this root directory:

```powershell
npx.cmd supabase login
npx.cmd supabase link --project-ref YOUR_PROJECT_REF
npx.cmd supabase db push --dry-run
npx.cmd supabase db push
```

In the hosted dashboard, set **Authentication → URL Configuration**:

- Site URL: `http://127.0.0.1:5173`
- Redirect URL: `http://127.0.0.1:5173/auth/callback`
- Recovery redirect: `http://127.0.0.1:5173/auth/callback?recovery=1`
- Add equivalent `localhost` entries if you use that hostname.

Signup supports email confirmation. The callback restores the authenticated session. Password reset uses an emailed recovery link. Configure your SMTP provider and production URLs before hosting the app.

## Backend configuration

See `backend/.env.example`. The existing environment supports the following settings:

| Setting | Purpose |
| --- | --- |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` or `SUPABASE_KEY` | Publishable or anon key; never a service-role key |
| `FRONTEND_URL` | Frontend origin for CORS and the default authentication callback; localhost in development, Vercel in the production profile |
| `QDRANT_URL` or `QDRANT_CLUSTER_ENDPOINT` | Required Qdrant Cloud endpoint with inference enabled; HTTPS defaults to port 443, explicit URL ports are preserved |
| `QDRANT_API_KEY` | Backend-only Qdrant credential |
| `GROQ_API_KEY` | Backend-only answer-generation credential |
| `GENERATION_MODEL` | Defaults to `openai/gpt-oss-120b` |
| `RESEARCH_ANSWER_INPUT_TOKENS`, `RESEARCH_ANSWER_MAX_TOKENS` | Answer prompt estimate and output reservation; defaults to 3,500 / 1,400, separately from embedding windows and retrieval candidate limits |
| `RESEARCH_GENERATION_RETRIES`, `RESEARCH_GENERATION_RETRY_WAIT` | Up to two transient retries; waits at most eight seconds per retry before returning a provider cooldown |
| `RESEARCH_QDRANT_COLLECTION` | Defaults to `research_rag_chunks`; legacy `QDRANT_COLLECTION` is intentionally ignored |
| `RESEARCH_STORAGE_BUCKET` | Defaults to `research-documents`; must match your migration |
| `RESEARCH_REDIRECT_URL` | Defaults to `${FRONTEND_URL}/auth/callback`; can be overridden explicitly |
| `OCR`, `OCR_LANGUAGE`, `TESSERACT_CMD` | Optional scanned PDF / embedded Word image text extraction |
| `RESEARCH_RERANK_ENABLED`, `RESEARCH_RERANK_MODEL` | Optional cross-encoder reranking; disabled by default, requires `requirements-local.txt` when enabled |
| `RESEARCH_CHILD_TOKENS`, `RESEARCH_PARENT_TOKENS` | Token ceilings, configured as 360 / 1,600 by default; child text is always capped at 254 content tokens for MiniLM |
| `RESEARCH_CONTEXT_TOKENS` | Retrieval context token ceiling, defaulting to 12,000; the search candidate limit is fixed at 40 |

Indexing always sends text as Qdrant `Document` vectors using `sentence-transformers/all-minilm-l6-v2`, and semantic/hybrid searches send a `Document` query, with `cloud_inference=True` on the client. Enable inference on your Qdrant Cloud cluster and use an API key with the required access. The app downloads only the small MiniLM tokenizer files for chunk/context token accounting, cached in `backend/.cache/tokenizers`; embedding inference runs in Qdrant Cloud. An inference failure is reported on the affected file.

MiniLM's 256-token context window is the input limit for each text being embedded, not a requests-per-minute allowance. Both chunks and cloud search queries reserve two special tokens and use at most 254 content tokens. Qdrant embeds/searches the source text; Groq generates the answer. An **answer-generation HTTP 429** therefore refers to Groq's token/request quotas. Short cooldowns use bounded retries; longer cooldowns return HTTP 429 with `Retry-After` and the wait time, and repeated requests share the same cooldown. Requests that exceed the provider's per-request allowance are reported separately. Account quotas still apply.

MiniLM embeddings have 384 dimensions. Previously uploaded sources queue background reindexing from their retained originals on the next question if they use an incompatible model or an outdated index. Model payload checks and search filters prevent mixing incompatible vectors. No Supabase migration is required. The former `QDRANT_ALL_MINILM_L6_V2` switch and local embedding settings are no longer used.

The optional cross-encoder remains a separate reranking model. When explicitly enabled with its optional dependencies installed, it prepares in a background thread on startup; answers use hybrid ranking while it loads or if it is unavailable. Its download is cached in `backend/.cache/reranker`. OCR needs the optional OCR dependencies and Tesseract executable; `pytesseract` alone is insufficient. Image interpretation is not implemented; OCR extracts text from images.

## Research workflow

1. Create an account and confirm your email, then log in.
2. Add PDF, DOCX, UTF-8 text, or Markdown sources, up to 25 MB each. File requests return after queuing background work. Two upload transfers and two indexing workers can run concurrently, with deletion handled separately. Each file shows its own stage and progress; chat, history, navigation, more uploads, and other ready files remain usable.
3. Select up to 30 ready sources. Selection is saved to the conversation when you submit a question.
4. Ask a question with hybrid, semantic, or text retrieval. Automatic scope searches every selected document for comparison questions; focused scope retrieves the closest passages.
5. Click an inline citation, citation chip, or source title to open the web PDF viewer. PDF citations open the original PDF at the cited page. Other supported file formats get a readable PDF text preview. Continue with follow-up questions; search, reopen, rename, or delete saved conversations in the sidebar.

Follow-up search context uses only the current conversation, without an extra generation request for rewriting. The answer provider returns JSON blocks with source IDs; the app validates those IDs against the actual retrieved excerpts and renders Markdown with clickable PDF citations. Valid Markdown citations are also accepted for compatibility. Malformed responses get one regeneration attempt; unknown source IDs are rejected rather than invented. Document excerpts are treated as untrusted source material in the generation prompt. Reference validation does not independently prove that each generated claim is supported. Review important claims in the source passages.

Failed indexing retains the original. Use **Retry indexing** to reuse it, or **Choose the file again** if an older failed upload no longer has an original. Processing files can be cancelled and removed. Deletion failures have a per-file retry action. Files with no completed index can be removed even when Qdrant is unavailable; scoped vector cleanup is saved locally and attempted separately. Processing, deleting, and failed files are excluded from chat source selection.

## Code layout and behavior

- `backend/app`: configuration, request schemas, Supabase adapter, extraction/chunking, vector store, answer-provider transport/retries, RAG orchestration, background file jobs, and HTTP routes.
- `backend/tests`: extraction, ownership filtering against local Qdrant, citation validation, authorization, atomic-save orchestration, file cancellation/retry, cloud API contracts, and idempotent retry checks.
- `frontend/src`: public/account pages, workspace components, authentication, typed API calls, and workspace state.
- `supabase`: local configuration and the application migration.

Every database request carries the user's access token, with Supabase row-level security enforcing ownership. Qdrant search and deletion always require both an owner and explicit document IDs. Original files use user-prefixed paths in a private bucket. Generation and vector credentials never reach the browser.

Each chat question and answer is committed together through `research_save_turn`. A request UUID makes retrying a response after a network failure idempotent. Failed generation writes no partial turn. Conversation timestamps and titles update when a turn saves. Deleting a document removes it from saved source selections while keeping historical cited excerpts.

Chunking is hierarchical and structure-aware: complete paragraphs, list entries, and sentences are grouped within PDF pages and Word/Markdown sections. Sparse TF-IDF vocabulary overlap identifies topic boundaries without embedding inference. Searchable child chunks use MiniLM's tokenizer with at most 254 content tokens, reserving two special tokens within its 256-token window. Only an oversized sentence/table row needs a word-boundary fallback. Larger parent passages retain up to 1,600 tokens of source context. Overlap uses whole sentences. These are ceilings, not fixed chunk lengths.

Hybrid retrieval combines dense cosine search and BM25 with reciprocal rank fusion. Each query requests the top 40 candidates from the selected documents, independent of `top_k` and document scope, and uses HNSW `ef=128`. Follow-up rankings are fused into at most 40 candidates before optional cross-encoder reranking. The app then deduplicates and expands parent contexts, selecting up to 8 by default for focused answers. The 12,000-token retrieval context ceiling remains available, while the answer prompt also obeys the separate 3,500-token input estimate by default, including the question, recent history, source metadata, and instructions. UTF-8 input estimates and the embedding tokenizer bound source windows; actual answer-provider token usage is logged without document content. Up to 1,400 completion tokens are reserved by default. Configure these budgets for your Groq account's allowance. Comparison/summary queries prioritize source coverage within the returned candidates; documents outside the top 40 are not guaranteed context. There is no arbitrary cosine cutoff that can discard a valid passage. Follow-ups search both the original question and the conversation-context query. Source-ID format failures get one regeneration attempt with the same bounded evidence, instead of being mislabeled as missing PDF evidence.

Previously uploaded sources are automatically queued for background reindexing from their saved originals when a question detects an outdated index. The question returns an actionable response immediately; ask using other ready sources or retry once indexing finishes. Index replacement writes new vectors before removing the old revision. Qdrant creates all required owner, document, version, model, and revision filter indexes, including when checking older collections. No SQL migration or manual re-upload is needed for this chunking upgrade. The lexical implementation scans only the selected, user-owned corpus and is capped at 30,000 chunks; use a sparse index for larger workloads. Uploads are limited to 500 PDF pages and 6,000 chunks.

File workers run inside a single backend process. Pending upload bytes and deferred vector cleanup are stored in `backend/.cache`; stale processing rows resume on the next library refresh using the signed-in user's current token. Keep that cache persistent across backend restarts. Use a durable external queue and coordinated locks before running multiple backend workers or replicas.

For other deployments, build the frontend with `npm.cmd run build`, serve `dist` with SPA fallback, and set `VITE_BACKEND_URL` to the backend origin before building. Configure backend `FRONTEND_URL`, callback/CORS URLs, and HTTPS. Keep provider secrets in backend environment variables.

## Verification

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
cd ..\frontend
npm.cmd test
npm.cmd run build
```

The tests run with local Qdrant, the real Qdrant SDK using mocked HTTP transport for cloud inference requests, mocked chat persistence/generation, and rendered frontend controls. Hosted authentication, Supabase SQL/storage policies, cloud inference availability, model downloads, and live generated answers require a migrated database and working network access; they are separate integration checks.
