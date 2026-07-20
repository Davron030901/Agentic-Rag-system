# Fully Free Deployment Runbook ($0/month)

This guide takes the project from an empty machine to a **live, publicly
reachable app** without spending anything. Follow it top to bottom; every step
has a verification you can run before moving on.

**Total cost: $0.** No credit card is required for any service below.

---

## The free stack

| Layer | Service | Free allowance | Catch |
|---|---|---|---|
| Frontend | **Vercel Hobby** | 100GB bandwidth, 1M function invocations, 100 build-min/month | Non-commercial use only |
| Backend | **Hugging Face Spaces** (Docker) | 2 vCPU, 16GB RAM, unlimited hosting | Sleeps after 48h idle; storage is ephemeral |
| Vector DB | **Qdrant Cloud Free** | 1GB RAM, 4GB disk, ~1M vectors @768-dim | Suspends after 1 week idle, deleted after 4 weeks |
| LLM + embeddings | **Google Gemini** free tier | `gemini-1.5-flash` + `text-embedding-004` | Rate-limited |
| Web search | **Tavily** free tier | Monthly search credit | Rate-limited |

> **Why Gemini and not OpenAI here?** OpenAI has no free tier — every call costs
> money. Gemini's free tier covers chat, vision captioning, and embeddings, which
> is everything this project needs. The code auto-detects the provider, so
> switching is just an environment variable.

> **Why Qdrant Cloud and not the embedded database?** Hugging Face Spaces gives
> you *ephemeral* storage: the container disk is wiped on every rebuild. With
> embedded Qdrant your entire index would vanish and you would have to re-upload
> all documents. Qdrant Cloud's free cluster persists it.

---

## Step 1 — Get the three API keys

### 1a. Google Gemini (LLM + embeddings + vision)
1. Go to **Google AI Studio** → <https://aistudio.google.com/app/apikey>.
2. Sign in with a Google account → **Create API key**.
3. Copy it. This is your `GOOGLE_API_KEY`.

### 1b. Tavily (web-search fallback)
1. Go to <https://tavily.com> → sign up (free).
2. Copy the API key from the dashboard. This is your `TAVILY_API_KEY`.

### 1c. Qdrant Cloud (vector database)
1. Go to <https://cloud.qdrant.io> → sign up (no credit card).
2. **Create cluster** → choose the **Free** tier → pick the region closest to you.
3. Wait ~1 minute for it to provision.
4. Copy two things:
   - The **cluster URL**, e.g. `https://abc123.eu-central.aws.cloud.qdrant.io:6333` → `QDRANT_URL`
   - The **API key** (create one under *Data Access Control* if not shown) → `QDRANT_API_KEY`

**Verify Qdrant is reachable:**
```bash
curl -H "api-key: YOUR_QDRANT_API_KEY" "YOUR_QDRANT_URL/collections"
# Expect: {"result":{"collections":[]},"status":"ok", ...}
```

---

## Step 2 — Test locally before deploying

Deploying broken code is painful to debug on a remote Space, so run it once on
your machine first.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` so it contains:

```env
GOOGLE_API_KEY=your_gemini_key
PROVIDER=gemini
TAVILY_API_KEY=your_tavily_key
QDRANT_URL=https://abc123.xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
CORS_ALLOW_ORIGINS=http://localhost:3000
```

Run it:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

**Verify:**
```bash
curl http://localhost:7860/health
# Expect: {"status":"ok","provider":"gemini","collection":"docs_gemini","embedding_dim":768,...}
```

If `provider` says `gemini` and `embedding_dim` is `768`, provider detection and
the Qdrant connection are both working.

Now upload one document and ask one question:
```bash
curl -F "file=@/path/to/any.pdf" http://localhost:7860/ingest
curl -s -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}'
```

You should get an answer with `steps` and `sources`. If so, you are ready to
deploy.

---

## Step 3 — Deploy the backend to Hugging Face Spaces

The `backend/` directory is already a valid Space: it has a `Dockerfile` that
listens on port **7860** and a `README.md` with the required frontmatter
(`sdk: docker`, `app_port: 7860`).

1. Create an account at <https://huggingface.co> (free).
2. **New** → **Space**. Fill in:
   - **Space name:** e.g. `agentic-rag`
   - **License:** your choice
   - **SDK:** **Docker** → **Blank**
   - **Visibility:** Public (private Spaces also work on free tier)
3. Push the **contents of `backend/`** to the Space repo root — not the whole
   monorepo, or the Dockerfile won't be at the root where Spaces expects it:

```bash
cd backend
git init
git add .
git commit -m "backend: adaptive agentic RAG"
git branch -M main
git remote add space https://huggingface.co/spaces/<your-username>/agentic-rag
git push space main
```

> Hugging Face will ask for credentials. Use your username and an **access token**
> (Settings → Access Tokens → New token with *write* permission) as the password.

4. In the Space, go to **Settings → Variables and secrets** and add these as
   **Secrets** (not public variables):

| Name | Value |
|---|---|
| `GOOGLE_API_KEY` | your Gemini key |
| `PROVIDER` | `gemini` |
| `TAVILY_API_KEY` | your Tavily key |
| `QDRANT_URL` | your Qdrant cluster URL |
| `QDRANT_API_KEY` | your Qdrant API key |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` for now — you'll update it in Step 5 |

5. The Space rebuilds automatically. Watch the **Logs** tab until it says the
   app is running.

**Verify (this is the public health URL from the checklist):**
```bash
curl https://<your-username>-agentic-rag.hf.space/health
# Expect HTTP 200 and {"status":"ok","provider":"gemini",...}
```

---

## Step 4 — Ingest your documents into the live backend

Your Qdrant Cloud cluster is shared between local and deployed environments, so
if you already ingested in Step 2 you can skip this. Otherwise upload against
the live Space:

```bash
curl -F "file=@/path/to/paper.pdf" \
  https://<your-username>-agentic-rag.hf.space/ingest
```

Expect a structured response like:
```json
{"filename":"paper.pdf","file_type":"pdf","chunks":48,"images_captioned":3,
 "pages":12,"status":"indexed","collection":"docs_gemini","provider":"gemini"}
```

You can also do this later through the frontend's upload UI — that's the point
of the uploader.

---

## Step 5 — Deploy the frontend to Vercel

1. Push the whole repository to GitHub (if you haven't already).
2. Go to <https://vercel.com> → sign up with GitHub → **Add New → Project** →
   import your repo.
3. **Important:** set **Root Directory** to `frontend`. Vercel will auto-detect
   Next.js.
4. Under **Environment Variables**, add:

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-username>-agentic-rag.hf.space` |

   No trailing slash.

5. **Deploy.** You'll get a URL like `https://agentic-rag-xyz.vercel.app`.

---

## Step 6 — Wire up CORS (the step everyone forgets)

Your browser will block the frontend from calling the backend until the backend
explicitly allows the Vercel origin.

1. Go back to your Space → **Settings → Variables and secrets**.
2. Edit `CORS_ALLOW_ORIGINS` to:
   ```
   https://agentic-rag-xyz.vercel.app,http://localhost:3000
   ```
   (your real Vercel URL, comma-separated, no spaces)
3. **Restart the Space** (Settings → Factory rebuild, or just Restart).

---

## Step 7 — Final verification checklist

Run through these in order. All must pass:

- [ ] `curl https://<user>-agentic-rag.hf.space/health` returns **200** with
      `"provider":"gemini"`.
- [ ] Opening the Vercel URL loads the chat UI.
- [ ] The chat input is **disabled** and shows *"Upload a document to get started"*.
- [ ] Uploading a PDF shows a spinner, then
      *"file.pdf — N pages, M chunks, K images captioned — indexed successfully."*
- [ ] The uploaded file appears in the **Indexed documents** list.
- [ ] The chat input is now **enabled**.
- [ ] Asking a document question returns an answer with **step pills**
      (`retrieve → grade docs → generate → grade generation`) and **citations**.
- [ ] Asking an out-of-document question (e.g. *"What is the capital of
      Australia?"*) triggers the **web search** pill and returns a web citation.
- [ ] Open browser DevTools → Network: no CORS errors.

If all boxes are checked, you have a fully deployed, fully free system.

---

## Step 8 — Keep the free tiers awake

Two free-tier timers can bite you:

- **Qdrant Cloud free cluster:** suspends after **1 week** of inactivity, and is
  **deleted after 4 weeks**. Losing this means losing your index.
- **Hugging Face Space:** sleeps after **48h** of inactivity (just a slow first
  request — nothing is lost).

This repo includes `.github/workflows/keepalive.yml`, a GitHub Actions cron job
that pings both twice a week. GitHub Actions is free for public repositories.

To enable it:
1. Push the repo to GitHub.
2. Go to **Settings → Secrets and variables → Actions → New repository secret**
   and add:
   - `SPACE_HEALTH_URL` = `https://<user>-agentic-rag.hf.space/health`
   - `QDRANT_URL` = your Qdrant cluster URL
   - `QDRANT_API_KEY` = your Qdrant API key
3. Go to the **Actions** tab and enable workflows if prompted.

You can also trigger it manually from the Actions tab ("Run workflow") to confirm
it works.

---

## Troubleshooting

**`/health` returns 500 or the Space won't start.**
Check the Space **Logs** tab. The most common cause is a missing secret — the app
fails fast with *"No LLM provider configured"* if neither `GOOGLE_API_KEY` nor
`OPENAI_API_KEY` is set.

**Frontend shows a network/CORS error.**
`CORS_ALLOW_ORIGINS` doesn't include your exact Vercel URL (check `https://`, no
trailing slash), or you didn't restart the Space after changing it.

**Answers are always "I don't know based on the available context."**
Nothing is indexed in the collection your provider is using. Confirm `/health`
shows the collection you ingested into. Remember: `docs_gemini` and `docs_openai`
are separate — if you ingested with OpenAI and now run Gemini, you must re-ingest.

**"Vector dimension error" from Qdrant.**
You pointed a Gemini deployment (768-dim) at a collection created by OpenAI
(1536-dim), or vice versa. Delete the wrong collection in the Qdrant Cloud
dashboard and re-ingest.

**429 / rate-limit errors under load.**
Expected on free tiers. One question makes several LLM calls (grade each chunk,
generate, then two generation graders). Reduce `TOP_K` to 2 and
`MAX_REGENERATIONS` to 1 to cut calls per query.

**First request after a day is very slow.**
The Space was asleep. This is normal free-tier cold start (20–60s). The keep-alive
workflow reduces how often it happens.

**Ingesting a large image-heavy PDF is slow or hits limits.**
Every embedded image triggers a vision call. Try a smaller PDF first, or accept
that ingestion of a 50-page image-heavy document takes a few minutes on free tier.

---

## Cost summary

| Service | Monthly cost |
|---|---|
| Vercel Hobby | $0 |
| Hugging Face Space (CPU basic) | $0 |
| Qdrant Cloud Free cluster | $0 |
| Gemini free tier | $0 |
| Tavily free tier | $0 |
| GitHub Actions (public repo) | $0 |
| **Total** | **$0** |

The only way this starts costing money is if you switch to `OPENAI_API_KEY`
(pay-per-token), upgrade the Space to GPU hardware, outgrow the Qdrant free
cluster, or move Vercel to Pro for commercial use.
