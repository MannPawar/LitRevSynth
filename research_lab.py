"""
MSBA Research Lab — 3-Phase Literature Engine v4
Cloud edition: Ollama (cloud) · Parallel Semantic Scholar · Streamlit Cloud
"""
 
import os, re, json, sqlite3, time, math, requests, logging
import concurrent.futures
from datetime import datetime
import streamlit as st
 
logging.basicConfig(level=logging.ERROR)
 
# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
S2_BASE        = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS      = "title,authors,year,abstract,url,citationCount"
DB_PATH        = "/tmp/research_papers.db"   # /tmp is writable on Streamlit Cloud
CURRENT_YEAR   = datetime.now().year
RECENT_CUTOFF  = 2022
FETCH_PER_CALL = 20
TOP_N          = 10
MIN_FINAL      = 8
MAX_WORKERS    = 8
 
# OpenRouter model options
MODEL_OPTIONS = {
    "google/gemini-3.1-flash-lite-preview (fast)":          "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen3.6-plus-preview          (free)":            "qwen/qwen3.6-plus-preview:free",
    "meta-llama/llama-3.3-70b-instruct  (fast · recommended)": "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1               (higher quality)":  "deepseek/deepseek-r1",
}
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Research Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ═══════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');
 
:root {
    --bg:     #0b0e14;
    --surf:   #111621;
    --surf2:  #161c2a;
    --border: #1e2740;
    --gold:   #c9a84c;
    --gold-d: #7a6130;
    --cyan:   #4ecdc4;
    --green:  #4caf84;
    --red:    #e05c5c;
    --text:   #d4d8e8;
    --dim:    #6b7394;
    --muted:  #3a4060;
}
 
/* ── hide sidebar collapse / expand arrow (all Streamlit versions) ── */
[data-testid="collapsedControl"],
[data-testid="baseButton-headerNoPadding"],
button[kind="headerNoPadding"],
[data-testid="stSidebarCollapsedControl"],
.st-emotion-cache-jnd7a1,
section[data-testid="stSidebar"] > div:first-child > button,
[data-testid="stSidebarNav"] ~ button {
    display: none !important;
}
 
/* ── global background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, section.main,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
div[class*="block-container"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
 
/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surf) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--dim) !important; }
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--gold) !important; }
 
/* ── typography ── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: var(--gold) !important;
}
p, li, span, label, div { font-family: 'Inter', sans-serif !important; }
 
/* ── text input ── */
[data-testid="stTextInput"] input {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }
 
/* ── primary button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg,#c9a84c,#9a7535) !important;
    color: #0b0e14 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.85 !important; }
 
/* ── secondary buttons ── */
[data-testid="stButton"] button {
    background: var(--surf2) !important;
    color: var(--dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--gold-d) !important;
    color: var(--gold) !important;
}
 
/* ── download button ── */
[data-testid="stDownloadButton"] button {
    background: var(--surf2) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(78,205,196,0.08) !important;
}
 
/* ── selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
 
/* ── progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,var(--gold-d),var(--gold)) !important;
}
[data-testid="stProgress"] { background: var(--surf2) !important; border-radius: 2px !important; }
 
/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    background: var(--surf) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
details summary {
    background: var(--surf) !important;
    color: var(--dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 0.75rem !important;
}
details summary:hover { color: var(--gold) !important; }
 
/* ── alerts: force readable on dark bg ── */
[data-testid="stAlert"], div[data-baseweb="notification"] {
    border-radius: 4px !important;
    border: 1px solid !important;
}
div[data-baseweb="notification"][kind="info"] {
    background: rgba(78,205,196,0.07) !important;
    border-color: var(--cyan) !important;
}
div[data-baseweb="notification"][kind="info"] *,
div[data-baseweb="notification"][kind="info"] p { color: var(--cyan) !important; }
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(76,175,132,0.07) !important;
    border-color: var(--green) !important;
}
div[data-baseweb="notification"][kind="positive"] *,
div[data-baseweb="notification"][kind="positive"] p { color: var(--green) !important; }
div[data-baseweb="notification"][kind="warning"] {
    background: rgba(201,168,76,0.07) !important;
    border-color: var(--gold) !important;
}
div[data-baseweb="notification"][kind="warning"] *,
div[data-baseweb="notification"][kind="warning"] p { color: var(--gold) !important; }
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(224,92,92,0.07) !important;
    border-color: var(--red) !important;
}
div[data-baseweb="notification"][kind="negative"] *,
div[data-baseweb="notification"][kind="negative"] p { color: var(--red) !important; }
 
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
 
/* ── custom layout components ── */
.lab-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.5rem !important;
    font-weight: 300 !important;
    color: var(--gold) !important;
    letter-spacing: 0.06em !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}
.lab-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}
.phase-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.stat-box {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.stat-n {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--gold);
    line-height: 1;
}
.stat-l {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.paper-card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold-d);
    border-radius: 0 5px 5px 0;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.4rem;
}
.paper-card.recent { border-left-color: var(--cyan); }
.ptitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text) !important;
    line-height: 1.3;
    display: block;
    margin-bottom: 0.18rem;
}
.pmeta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: var(--muted) !important;
}
.pmeta .cy { color: var(--gold-d) !important; }
.pmeta .yr { color: var(--dim) !important; }
.pmeta .tr { color: var(--cyan) !important; }
.pmeta .ts { color: var(--gold-d) !important; }
.rbadge {
    display: inline-block;
    background: rgba(201,168,76,0.1);
    border: 1px solid var(--gold-d);
    color: var(--gold) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    padding: 1px 5px;
    border-radius: 3px;
    margin-right: 0.35rem;
    letter-spacing: 0.04em;
}
 
/* ── review essay typography ── */
[data-testid="stMarkdown"] h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.85rem !important;
    font-weight: 400 !important;
    color: var(--gold) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1.2rem !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stMarkdown"] h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin-top: 1.8rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMarkdown"] h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--dim) !important;
    margin-top: 1.2rem !important;
}
[data-testid="stMarkdown"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.82 !important;
    color: var(--text) !important;
}
[data-testid="stMarkdown"] li {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}
[data-testid="stMarkdown"] strong {
    color: var(--gold) !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdown"] em {
    color: var(--dim) !important;
    font-style: italic !important;
}
[data-testid="stMarkdown"] a {
    color: var(--cyan) !important;
    text-decoration: none !important;
}
[data-testid="stMarkdown"] a:hover { text-decoration: underline !important; }
[data-testid="stMarkdown"] blockquote {
    border-left: 3px solid var(--gold-d) !important;
    padding-left: 1rem !important;
    color: var(--dim) !important;
    font-style: italic !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── session state ─────────────────────────────────────────────────
for k, v in {
    "papers_all": [], "papers_top": [], "review": "",
    "history": [], "running": False, "elapsed": 0,
    "selected_model": DEFAULT_MODEL,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
 
# ═══════════════════════════════════════════════════════════════════
#  OPENROUTER CLIENT  (OpenAI-compatible REST endpoint)
# ═══════════════════════════════════════════════════════════════════
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
 
def get_api_key() -> str:
    """
    Resolve the OpenRouter API key.
    Priority: st.secrets → environment variable → sidebar input.
    """
    try:
        return st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")
 
 
def stream_synthesis(api_key: str, model_name: str, prompt: str):
    """Stream OpenRouter /v1/chat/completions response, yielding text chunks."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://research-lab.streamlit.app",
        "X-Title": "MSBA Research Lab",
    }
    payload = {
        "model":  model_name,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    with requests.post(
        f"{OPENROUTER_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() in ("", "[DONE]"):
                continue
            try:
                data = json.loads(line)
                delta = data["choices"][0].get("delta", {})
                text  = delta.get("content") or ""
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
 
 
def fallback_synthesis(api_key: str, model_name: str, topic: str, papers: list) -> str:
    """Non-streaming fallback with a simpler prompt."""
    summary = "\n".join(
        f"{i+1}. {p['title']} — {p['authors'][0] if p['authors'] else '?'} "
        f"({p['year']}) [{p['citations']} citations]"
        for i, p in enumerate(papers))
    prompt = (
        f'Write a literature review on "{topic}".\n\nPapers:\n{summary}\n\n'
        f"Sections: Introduction, Key Contributions, Research Gaps, References.\n"
        f"Start with '# Literature Review: {topic}'. Academic prose. Min 500 words."
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://research-lab.streamlit.app",
        "X-Title": "MSBA Research Lab",
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model":  model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"# Literature Review: {topic}\n\nSynthesis failed: {e}"
 
 
# ═══════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER,
        citations INTEGER, abstract TEXT, url TEXT, source_tag TEXT,
        score REAL DEFAULT 0, topic TEXT)""")
    conn.commit()
    return conn
 
 
def db_batch_upsert(conn: sqlite3.Connection, papers: list, topic: str):
    conn.executemany(
        """INSERT OR REPLACE INTO papers
           (id,title,authors,year,citations,abstract,url,source_tag,topic)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [(p["id"], p["title"], json.dumps(p["authors"]), p["year"],
          p["citations"], p["abstract"], p["url"], p["source_tag"], topic)
         for p in papers])
    conn.commit()
 
 
# ═══════════════════════════════════════════════════════════════════
#  PHASE 1 — PARALLEL HARVEST
# ═══════════════════════════════════════════════════════════════════
ANGLE_TEMPLATES = [
    "{t}",
    "{t} methods algorithms techniques",
    "{t} survey review overview",
    "{t} applications case study",
    "{t} deep learning neural network",
    "{t} evaluation benchmark performance",
    "{t} challenges open problems",
    "{t} framework system design",
]
 
 
def _make_jobs(topic: str) -> list:
    jobs = []
    for tpl in ANGLE_TEMPLATES:
        q = tpl.format(t=topic)
        jobs.append((q, ""))
        jobs.append((q, f"&year={RECENT_CUTOFF}-{CURRENT_YEAR}"))
    return jobs
 
 
def _fetch_one(args: tuple) -> list:
    query, extra = args
    url = (f"{S2_BASE}?query={requests.utils.quote(query)}"
           f"&limit={FETCH_PER_CALL}&fields={S2_FIELDS}{extra}")
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "ResearchLab/4.0"}, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json().get("data", [])
            return []
        except Exception:
            time.sleep(1)
    return []
 
 
def harvest_parallel(topic: str) -> list:
    jobs = _make_jobs(topic)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        batches = list(pool.map(_fetch_one, jobs))
 
    seen: set = set()
    papers: list = []
    for batch in batches:
        for item in batch:
            pid = item.get("paperId") or item.get("url") or ""
            if not pid or pid in seen or not item.get("abstract"):
                continue
            seen.add(pid)
            year = item.get("year") or 0
            papers.append({
                "id":        pid,
                "title":     (item.get("title") or "Untitled").strip(),
                "authors":   [a["name"] for a in item.get("authors", [])],
                "year":      year,
                "citations": item.get("citationCount") or 0,
                "abstract":  (item.get("abstract") or "").strip(),
                "url":       item.get("url") or "",
                "source_tag": "RECENT" if year >= RECENT_CUTOFF else "SEMINAL",
            })
    return papers
 
 
# ═══════════════════════════════════════════════════════════════════
#  PHASE 2 — SCORE & SELECT TOP N
# ═══════════════════════════════════════════════════════════════════
def score_paper(p: dict, tw: set) -> float:
    titw = set(re.findall(r"\w+", p["title"].lower()))
    absw = set(re.findall(r"\w+", p["abstract"].lower()))
    n = max(len(tw), 1)
    overlap = len(tw & titw) / n * 2.5 + len(tw & absw) / n * 1.0
    cite    = math.log(p["citations"] + 1)
    rec     = (1.5 if p["year"] >= RECENT_CUTOFF else
               0.5 if p["year"] >= RECENT_CUTOFF - 5 else 0.0)
    return cite + overlap * 3.0 + rec
 
 
def rank_and_select(papers: list, topic: str) -> list:
    tw = set(re.findall(r"\w+", topic.lower()))
    pool = papers
    for floor in (5, 1, 0):
        pool = [p for p in papers if p.get("abstract") and p.get("citations", 0) >= floor]
        if len(pool) >= MIN_FINAL:
            break
 
    for p in pool:
        p["score"] = score_paper(p, tw)
    scored  = sorted(pool, key=lambda x: x["score"], reverse=True)
    seminal = [p for p in scored if p["source_tag"] == "SEMINAL"]
    recent  = [p for p in scored if p["source_tag"] == "RECENT"]
 
    n_rec = max(min(round(TOP_N * 0.4), len(recent)),  min(2, len(recent)))
    n_sem = max(min(TOP_N - n_rec,      len(seminal)), min(2, len(seminal)))
    sel   = seminal[:n_sem] + recent[:n_rec]
    have  = {p["id"] for p in sel}
    sel  += [p for p in scored if p["id"] not in have][:TOP_N - len(sel)]
    have  = {p["id"] for p in sel}
    sel  += [p for p in scored if p["id"] not in have][:max(0, MIN_FINAL - len(sel))]
    return sel[:max(TOP_N, MIN_FINAL)]
 
 
# ═══════════════════════════════════════════════════════════════════
#  PHASE 3 — SYNTHESIS  (Ollama cloud API)
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a distinguished academic researcher writing formal literature reviews. "
    "You write ONLY in flowing Markdown prose. You NEVER output JSON or raw data. "
    "Your response always starts with a Markdown H1 heading."
)
 
 
def build_prompt(topic: str, papers: list) -> str:
    papers_block = ""
    for i, p in enumerate(papers, 1):
        auth = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        papers_block += (
            f"\n--- PAPER {i} ---\n"
            f"Title:     {p['title']}\n"
            f"Authors:   {auth}\n"
            f"Year:      {p['year']}\n"
            f"Citations: {p['citations']}\n"
            f"URL:       {p['url']}\n"
            f"Abstract:\n{p['abstract']}\n"
        )
 
    return (
        f'You are writing a formal academic literature review on: "{topic}"\n\n'
        f'You have been given {len(papers)} carefully selected papers. '
        f'Read every abstract thoroughly before writing.\n'
        f'{papers_block}\n'
        f'{"=" * 55}\n'
        f'WRITING INSTRUCTIONS — FOLLOW EXACTLY\n'
        f'{"=" * 55}\n\n'
        f'Write a complete academic literature review in Markdown.\n'
        f'Your output MUST follow this exact structure:\n\n'
        f'# Literature Review: {topic}\n\n'
        f'## 1. Introduction\n'
        f'Write 2–3 paragraphs introducing the field, its academic significance,\n'
        f'and the scope of this review. Define key terms where needed.\n\n'
        f'## 2. Foundational Work\n'
        f'Write 3–4 paragraphs on the seminal contributions.\n'
        f'For every paper discussed add an in-text citation: (Author et al., Year)\n'
        f'or (Author & Author, Year) for two authors.\n'
        f'Cover: what problem each paper solved · the method used · the key findings.\n\n'
        f'## 3. Recent Advancements ({RECENT_CUTOFF}–{CURRENT_YEAR})\n'
        f'Write 3–4 paragraphs on the more recent papers.\n'
        f'Explain how each extends, challenges, or refines prior work.\n'
        f'Use in-text citations for every paper mentioned.\n\n'
        f'## 4. Synthesis and Critical Analysis\n'
        f'Write 2–3 paragraphs comparing methodologies across papers.\n'
        f'Identify tensions, contradictions, or open debates in the literature.\n\n'
        f'## 5. Research Gaps and Future Directions\n'
        f'Write 2–3 paragraphs on what the literature has NOT yet addressed.\n'
        f'Name each gap, explain why it matters, and what work could fill it.\n'
        f'Weave at least three numbered gaps into the prose:\n'
        f'  "First, ... Second, ... Third, ..."\n\n'
        f'## 6. Conclusion\n'
        f'One paragraph summarising the current state of the field and its trajectory.\n\n'
        f'## References\n'
        f'List ALL cited papers in APA 7th edition format, one per line.\n'
        f'Example: Smith, J., & Jones, A. (2021). Title. *Journal*, *3*(2), 1–10. https://doi.org/...\n\n'
        f'{"=" * 55}\n'
        f'CITATION RULES — MANDATORY:\n'
        f'- Cite EVERY one of the {len(papers)} papers at least once in the body\n'
        f'- (First Author et al., Year) for 3+ authors; (A & B, Year) for 2; (A, Year) for 1\n'
        f'- NEVER fabricate citations — only cite the {len(papers)} papers listed above\n'
        f'- Every in-text citation must appear in the References section\n\n'
        f'OUTPUT RULES — MANDATORY:\n'
        f'- Start IMMEDIATELY with "# Literature Review: {topic}" — no preamble\n'
        f'- Write ONLY flowing academic prose — no JSON, bullet dumps, or raw data\n'
        f'- Minimum 900 words total\n'
        f'{"=" * 55}\n\n'
        f'Begin writing the literature review now:'
    )
 
 
def is_essay(text: str) -> bool:
    s = text.strip()
    return s.startswith("#") and not s.startswith("{") and len(s.split()) > 150
 
 
def save_bibtex(papers: list) -> str:
    lines = []
    for p in papers:
        last = p["authors"][0].split()[-1] if p["authors"] else "Unknown"
        lines.append(
            f"@article{{{last}{p['year']},\n  title  = {{{p['title']}}},\n"
            f"  author = {{{', '.join(p['authors'][:3])}}},\n"
            f"  year   = {{{p['year']}}},\n  url    = {{{p['url']}}},\n"
            f"  note   = {{{p['citations']} citations}}\n}}")
    content = "\n\n".join(lines)
    with open("/tmp/references.bib", "w", encoding="utf-8") as f:
        f.write(content)
    return content
 
 
# ═══════════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════
def stat_html(n, label: str) -> str:
    return (f'<div class="stat-box"><div class="stat-n">{n}</div>'
            f'<div class="stat-l">{label}</div></div>')
 
 
def card_html(p: dict, rank: int = None) -> str:
    tag   = "recent" if p["source_tag"] == "RECENT" else "seminal"
    badge = f'<span class="rbadge">#{rank}</span>' if rank else ""
    tspan = ('<span class="tr">RECENT</span>' if tag == "recent"
             else '<span class="ts">SEMINAL</span>')
    auth  = p["authors"][0][:28] if p["authors"] else "Unknown"
    return (
        f'<div class="paper-card {tag}">'
        f'{badge}<span class="ptitle">{p["title"][:88]}</span>'
        f'<div class="pmeta">'
        f'<span class="yr">{p["year"]}</span> '
        f'<span class="cy">▲ {p["citations"]:,}</span> '
        f'{tspan} <span>{auth}</span>'
        f'</div></div>'
    )
 
 
# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.35rem;'
        'color:#c9a84c;letter-spacing:0.1em;margin-bottom:0">◈ RESEARCH LAB</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
        'color:#3a4060;letter-spacing:0.22em;text-transform:uppercase;margin-top:0.1rem">'
        'MSBA · Literature Engine v4</p>',
        unsafe_allow_html=True)
    st.divider()
 
    # ── API key ───────────────────────────────────────────────────
    api_key = get_api_key()
    if not api_key:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#e05c5c;letter-spacing:0.1em;text-transform:uppercase">'
            '⚠ API Key Required</p>', unsafe_allow_html=True)
        api_key = st.text_input(
            "OpenRouter API Key", type="password",
            placeholder="sk-or-...",
            help="Get yours at openrouter.ai/keys",
            label_visibility="collapsed")
        if api_key:
            st.success("Key entered ✓")
        else:
            st.caption("Set OPENROUTER_API_KEY in Streamlit secrets or enter above.")
    else:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#4caf84;letter-spacing:0.1em">● API key loaded</p>',
            unsafe_allow_html=True)
 
    st.divider()
 
    # ── Model selector ─────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
        'color:#6b7394;letter-spacing:0.14em;text-transform:uppercase">Model</p>',
        unsafe_allow_html=True)
    model_label = st.selectbox(
        "model", list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed")
    selected_model = MODEL_OPTIONS[model_label]
    st.session_state.selected_model = selected_model
 
    st.divider()
 
    # ── History ───────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
        'color:#6b7394;letter-spacing:0.14em;text-transform:uppercase">History</p>',
        unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            'color:#3a4060">no previous runs</p>', unsafe_allow_html=True)
    for i, h in enumerate(reversed(st.session_state.history)):
        if st.button(f"  {h['topic'][:26]}…", key=f"h{i}", use_container_width=True):
            st.session_state.review     = h["review"]
            st.session_state.papers_top = h["top"]
            st.session_state.papers_all = h.get("all", [])
            st.session_state.elapsed    = h.get("elapsed", 0)
 
 
# ═══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
st.markdown('<h1 class="lab-title">Literature Review Engine</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lab-sub">Semantic Scholar · OpenRouter · Streamlit Edition</p>',
    unsafe_allow_html=True)
 
ic, bc = st.columns([5, 1])
with ic:
    topic = st.text_input(
        "t", label_visibility="collapsed",
        placeholder="Research field  ·  e.g.  PCG in Games  ·  Process Mining  ·  Federated Learning")
with bc:
    run = st.button(
        "▶  Generate", type="primary", use_container_width=True,
        disabled=(st.session_state.running or not api_key))
 
st.divider()
 
# phase strip
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="phase-label">01 · HARVEST</div>', unsafe_allow_html=True)
    b1 = st.empty(); e1 = st.container()
with c2:
    st.markdown('<div class="phase-label">02 · RANK</div>', unsafe_allow_html=True)
    b2 = st.empty(); e2 = st.container()
with c3:
    st.markdown('<div class="phase-label">03 · SYNTHESIZE</div>', unsafe_allow_html=True)
    b3 = st.empty()
 
st.divider()
 
# stat strip
s1, s2, s3, s4 = st.columns(4)
sh  = s1.empty()
ss  = s2.empty()
sr  = s3.empty()
st_ = s4.empty()
 
st.divider()
review_slot = st.empty()
dl_slot     = st.empty()
 
# restore on history navigation
if st.session_state.review and not st.session_state.running:
    review_slot.markdown(st.session_state.review)
 
if st.session_state.papers_top:
    tp = st.session_state.papers_top
    sh.markdown(stat_html(len(st.session_state.papers_all), "HARVESTED"), unsafe_allow_html=True)
    ss.markdown(stat_html(sum(1 for p in tp if p["source_tag"] == "SEMINAL"), "SEMINAL"), unsafe_allow_html=True)
    sr.markdown(stat_html(sum(1 for p in tp if p["source_tag"] == "RECENT"),  "RECENT"),  unsafe_allow_html=True)
    st_.markdown(stat_html(st.session_state.elapsed, "SECONDS"), unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════
if run and topic.strip() and api_key:
    topic  = topic.strip()
    model  = st.session_state.selected_model
 
    st.session_state.running = True
    st.session_state.review  = ""
    t0 = time.time()
 
    sh.markdown(stat_html("…", "HARVESTED"), unsafe_allow_html=True)
    ss.markdown(stat_html("…", "SEMINAL"),   unsafe_allow_html=True)
    sr.markdown(stat_html("…", "RECENT"),    unsafe_allow_html=True)
    st_.markdown(stat_html("…", "SECONDS"),  unsafe_allow_html=True)
 
    # ── Phase 1: Harvest ─────────────────────────────────────────
    b1.info("Scanning Semantic Scholar in parallel…")
    papers_all = harvest_parallel(topic)
 
    if not papers_all:
        b1.error("No papers found — check topic spelling or try a broader term.")
        st.session_state.running = False
        st.stop()
 
    conn = db_connect()
    db_batch_upsert(conn, papers_all, topic)
    st.session_state.papers_all = papers_all
 
    n_s = sum(1 for p in papers_all if p["source_tag"] == "SEMINAL")
    n_r = sum(1 for p in papers_all if p["source_tag"] == "RECENT")
    b1.success(f"Harvested {len(papers_all)} unique papers")
    sh.markdown(stat_html(len(papers_all), "HARVESTED"), unsafe_allow_html=True)
    ss.markdown(stat_html(n_s, "SEMINAL"), unsafe_allow_html=True)
    sr.markdown(stat_html(n_r, "RECENT"),  unsafe_allow_html=True)
 
    with e1.expander(f"All {len(papers_all)} papers", expanded=False):
        st.markdown(
            "".join(card_html(p) for p in
                    sorted(papers_all, key=lambda x: x["citations"], reverse=True)),
            unsafe_allow_html=True)
 
    # ── Phase 2: Rank ─────────────────────────────────────────────
    b2.info("Scoring and ranking…")
    papers_top = rank_and_select(papers_all, topic)
    st.session_state.papers_top = papers_top
 
    for p in papers_top:
        conn.execute("UPDATE papers SET score=? WHERE id=?", (p["score"], p["id"]))
    conn.commit()
 
    b2.success(f"Selected top {len(papers_top)} papers")
    with e2.expander(f"Top {len(papers_top)} selected", expanded=True):
        st.markdown(
            "".join(card_html(p, rank=i + 1) for i, p in enumerate(papers_top)),
            unsafe_allow_html=True)
 
    # ── Phase 3: Synthesize ───────────────────────────────────────
    model_display = model.split(":")[0].replace("-", " ").title()
    b3.info(f"Synthesizing with {model_display}…")
    prompt = build_prompt(topic, papers_top)
 
    accumulated = ""
    live = review_slot.empty()
 
    try:
        for chunk in stream_synthesis(api_key, model, prompt):
            accumulated += chunk
            live.markdown(accumulated + "▌")
        live.markdown(accumulated)
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
            b3.error("Invalid API key — check your OpenRouter key in the sidebar.")
            st.session_state.running = False
            st.stop()
        elif "quota" in err_str or "rate" in err_str or "429" in err_str:
            b3.warning("Rate limit hit — running fallback (non-streaming)…")
            accumulated = fallback_synthesis(api_key, model, topic, papers_top)
            review_slot.markdown(accumulated)
        else:
            b3.warning(f"Streaming error ({e}) — running fallback…")
            accumulated = fallback_synthesis(api_key, model, topic, papers_top)
            review_slot.markdown(accumulated)
 
    if not is_essay(accumulated):
        b3.warning("Output malformed — running fallback…")
        accumulated = fallback_synthesis(api_key, model, topic, papers_top)
        review_slot.markdown(accumulated)
 
    st.session_state.review = accumulated
    bib_content = save_bibtex(papers_top)
 
    elapsed = int(time.time() - t0)
    st.session_state.elapsed = elapsed
    st_.markdown(stat_html(elapsed, "SECONDS"), unsafe_allow_html=True)
    b3.success(f"Complete — {elapsed}s")
 
    st.session_state.history.append({
        "topic": topic, "review": accumulated,
        "top": papers_top, "all": papers_all, "elapsed": elapsed,
    })
    st.session_state.running = False
 
    with dl_slot.container():
        st.divider()
        da, db_ = st.columns(2)
        da.download_button(
            "↓  Download Review (.md)", data=accumulated,
            file_name=f"review_{topic[:30]}.md", mime="text/markdown")
        db_.download_button(
            "↓  Download BibTeX", data=bib_content,
            file_name="references.bib")
 
elif run and not topic.strip():
    b1.warning("Please enter a research topic.")
 
elif run and not api_key:
    st.error("Enter your OpenRouter API key in the sidebar first.")
