# STUDY.md — understand jd-matcher properly (picture first)

Read this before you claim the project in an interview. The goal: you can
explain every part in your own words, not repeat these sentences.

## The picture first

You are applying to jobs. Each job post is a wall of text, and your resume is
another wall of text. **jd-matcher answers one question: "which of these jobs
is closest to what I already am, and what exactly am I missing for each one?"**

It does that in two independent ways, and you must be able to explain both:

1. **Semantic match (embeddings + cosine similarity)** — how close the *meaning*
   of your resume is to the *meaning* of each job description.
2. **Skill-gap analysis (keyword matching)** — which named skills each JD asks
   for, which of those you already have, and which you are missing.

The final output is a ranked table plus one paragraph of advice per job.

## What is an embedding? (plain words)

An embedding is a list of numbers that represents the *meaning* of a piece of
text. This project uses `all-MiniLM-L6-v2`, which turns any text into a list
of **384 numbers**. Two texts with similar meaning get similar lists of
numbers; two texts about different things get different lists.

Think of it like a fingerprint for meaning. "Python developer who builds ML
models" and "Machine learning engineer, Python, PyTorch" have similar
fingerprints. "Python developer" and "Kubernetes platform engineer" do not —
even though both are tech jobs.

Why 384? That is just the size this particular model outputs. Other models
output 768, 1024, or more. Bigger is not automatically better; it is a
trade-off between detail and speed/memory.

## What is cosine similarity? (plain words)

Once you have two lists of numbers, you need one number that says "how
similar are these two?" Cosine similarity measures the **angle** between the
two lists (treated as arrows in 384-dimensional space).

- Angle = 0° (arrows point the same way) → similarity **1.0** → same meaning.
- Angle = 90° (arrows perpendicular) → similarity **0.0** → unrelated.
- In this project the score is clipped to [0, 1] and shown as 0–100.

Why the angle and not the distance? Because the *direction* of the arrow is
the meaning; the *length* mostly reflects how long the text is. A long resume
and a short resume about the same skills should still match. Cosine similarity
ignores length and compares direction only.

## What does RAG have to do with this?

This project is **not** a full RAG system — it has no LLM answering questions.
But it uses the exact same retrieval step that RAG is built on:

```
RAG pipeline:   ingest -> chunk -> embed -> store in vector DB
                -> embed the query -> cosine search -> top chunks -> LLM

jd-matcher:     ingest JDs + resume -> embed both
                -> cosine similarity resume vs each JD -> ranked list
```

So: embeddings + cosine similarity *is* the retrieval half of RAG. If you
understand this project, you understand the retrieval layer of CodeRAG too.
The difference is that CodeRAG goes one step further and hands the retrieved
chunks to an LLM to write an answer.

## How the skill-gap analysis works

Embeddings tell you "these texts are close in meaning" but they cannot tell
you *which skill* is missing. So the project does a second, much simpler pass:

1. `skills.yaml` holds ~30 skills. Each skill has keywords, e.g. PyTorch has
   keywords `pytorch` and `torch`.
2. The code lowercases the JD text and searches for each keyword with a
   **word-boundary regex** (`\bkeyword\b`). That is why `git` does not match
   inside `github`, and `ml` does not match inside `html`.
3. Skills found in the JD = **required skills**. Skills found in the resume =
   skills you have. **Missing = required − has.**
4. The report lists the top 3 missing skills per job and suggests closing the
   top gap with one small project before applying.

Limitations you should admit in an interview if asked: keyword matching is
dumb — it cannot tell that "built ML models" implies machine learning if the
words "machine learning" never appear, and it cannot detect synonyms that are
not in the YAML file. The embedding score covers the meaning; the keyword pass
covers the named skills. They complement each other.

## How the pieces fit together (the flow)

```
data/sample_resume.md ─┐
                       ├─> embedder (MiniLM, or TF-IDF fallback)
data/jds/*.md ─────────┘
                              │
                    cosine similarity ──> match score per JD
                              │
skills.yaml ──> keyword match ──> required / matched / missing skills
                              │
                              └─> ranked table + results/report.md
```

## 5 interview questions you should be able to answer

1. **"Walk me through this project."**
   It ranks job descriptions against a resume using embeddings and cosine
   similarity, then runs a keyword-based skill-gap analysis from a YAML skill
   list and writes a ranked report with one suggestion per job.
2. **"Why embeddings instead of just counting shared words?"**
   Counting shared words (TF-IDF) only sees exact overlap. Embeddings capture
   meaning: "built ML models" and "machine learning engineer" are close even
   with few shared words. That is why MiniLM is the default and TF-IDF is
   only the fallback.
3. **"Why cosine similarity and not Euclidean distance?"**
   Cosine compares direction, not length, so a long resume and a short JD
   about the same skills still match. Euclidean distance would be dominated
   by how much text there is.
4. **"What breaks, and what would you improve?"**
   Keyword matching misses synonyms and implied skills; the sample is only 5
   JDs; scores are relative, not absolute. Improvements: a bigger skill list
   with synonyms, a cross-encoder reranker on the top results, real JDs
   scraped or pasted in, and an LLM-written suggestion per job instead of a
   template sentence.
5. **"What does this have to do with RAG?"**
   It is the retrieval half of RAG — embed documents, embed the query,
   cosine-search the top matches. CodeRAG adds chunking, a vector database
   (pgvector), and an LLM that answers from the retrieved chunks.
