# jd-matcher

Rank job descriptions against your resume and explain the skill gaps.

## What it does

- Ingests job descriptions from `data/jds/` (5 realistic sample JDs included).
- Embeds your resume and each JD with **sentence-transformers
  `all-MiniLM-L6-v2`** (384-dim) and ranks the JDs by **cosine similarity**.
  If the model cannot be loaded, it falls back to **TF-IDF** vectors
  (scikit-learn) and says so — see "Embedding method" below.
- Runs a **skill-gap analysis** from `skills.yaml` (~30 AI/data skills):
  which required skills each JD lists, which ones your resume already has,
  and which are missing.
- Prints a ranked table and writes `results/report.md` with the same table
  plus a one-paragraph suggestion per job.

## How it works

```
resume + JDs ──> embeddings (MiniLM, or TF-IDF fallback)
             ──> cosine similarity ──> match score (0-100) per JD
skills.yaml  ──> keyword match ──> required / matched / missing skills
             ──> ranked table + results/report.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from
Hugging Face. That needs internet access once; after that it is cached.

## How to run

```bash
# Rank all JDs and write results/report.md
python main.py match

# Detail for one ranked job (e.g. the #2 job in the table)
python main.py gaps --job 2

# Use your own resume / JD folder
python main.py --resume /path/to/resume.md --jds /path/to/jds match
```

## Example output

```
Rank Role                                      Company           Score  Top missing skills
--------------------------------------------------------------------------------------------------------------
1     Junior AI Engineer                       TechNova Labs      69.1  none
2     Machine Learning Engineer (Entry Level)  DataBridge Inc     66.2  none
3     Generative AI Engineer                   PromptWorks        65.4  LangGraph
4     Backend Python Developer                 CloudServe         56.1  none
5     Junior Data Scientist                    Insight Analytic   54.3  Data Visualization
```

## Embedding method

This project tries MiniLM first and falls back to TF-IDF only if the model
cannot be loaded. **Verified on this machine (2026-11-23): MiniLM was
actually used** — `sentence-transformers/all-MiniLM-L6-v2` loaded and
produced the scores above. The TF-IDF fallback was exercised by the test
suite but was not needed for the real run.

## Tests

```bash
python -m pytest tests/ -v
```

7 tests: ranking order on controlled input, gap detection, empty JD folder,
report creation, cosine similarity bounds, TF-IDF fallback path,
word-boundary skill matching, and nice-to-have skills not counted as missing.

## Files

| File | What it is |
|------|------------|
| `main.py` | CLI + ranking + gap analysis |
| `skills.yaml` | ~30 skills with match keywords |
| `data/sample_resume.md` | Fictional fresher MS-AI resume (sample data) |
| `data/jds/` | 5 sample JDs (AI engineer, ML engineer, data scientist, GenAI engineer, backend Python dev) |
| `tests/test_main.py` | pytest suite |
| `STUDY.md` | Picture-first explainer: embeddings, cosine similarity, RAG, gaps, interview questions |
| `requirements.txt` | Dependencies |
