"""jd-matcher: rank job descriptions against a resume and explain skill gaps.

How it works
------------
1. The resume and every job description (JD) in data/jds/ are turned into
   vectors (embeddings) with sentence-transformers all-MiniLM-L6-v2
   (384 dimensions). If that model cannot be loaded (no internet, no
   torch, download failure), it falls back to TF-IDF vectors instead.
2. Cosine similarity between the resume vector and each JD vector gives a
   match score in [0, 1] (shown as 0-100).
3. Skill-gap analysis is separate and keyword-based: the skills in
   skills.yaml are looked up in each JD (required skills) and in the resume
   (skills the candidate has). Missing = required - has.

Usage
-----
    python main.py match            # rank all JDs, print table, write results/report.md
    python main.py gaps --job 2     # detail for the job ranked #2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JDS_DIR = os.path.join(PROJECT_ROOT, "data", "jds")
DEFAULT_RESUME = os.path.join(PROJECT_ROOT, "data", "sample_resume.md")
DEFAULT_SKILLS = os.path.join(PROJECT_ROOT, "skills.yaml")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "results", "report.md")

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_skills(path: str = DEFAULT_SKILLS) -> list[dict]:
    """Load the skills list from skills.yaml."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["skills"]


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@dataclass
class Job:
    title: str
    company: str
    location: str
    text: str
    filename: str
    required_text: str = ""
    score: float = 0.0
    required_skills: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


def parse_jd(path: str) -> Job:
    """Parse one JD markdown file. Title from the first '# ' line,
    company/location from 'Company:' / 'Location:' lines. The text under
    the '## Required Skills' heading (if present) is kept separately so
    the gap analysis only counts required skills -- 'Nice to Have' items
    are not treated as missing."""
    text = read_text(path)
    title, company, location = "Untitled role", "Unknown", "Unknown"
    required_lines: list[str] = []
    in_required = False
    for line in text.splitlines():
        if line.startswith("# ") and title == "Untitled role":
            title = line[2:].strip()
        elif line.lower().startswith("company:"):
            company = line.split(":", 1)[1].strip()
        elif line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
        if line.strip().lower() == "## required skills":
            in_required = True
            continue
        if in_required:
            if line.startswith("## "):
                in_required = False
            else:
                required_lines.append(line)
    required_text = "\n".join(required_lines).strip() or text
    return Job(
        title=title,
        company=company,
        location=location,
        text=text,
        filename=os.path.basename(path),
        required_text=required_text,
    )


def load_jobs(jds_dir: str = DEFAULT_JDS_DIR) -> list[Job]:
    """Load all .md files in the JDs folder. Returns [] if the folder
    is missing or empty (handled gracefully by the caller)."""
    if not os.path.isdir(jds_dir):
        return []
    jobs = []
    for name in sorted(os.listdir(jds_dir)):
        if name.lower().endswith(".md"):
            jobs.append(parse_jd(os.path.join(jds_dir, name)))
    return jobs


# ---------------------------------------------------------------------------
# Skill-gap analysis (keyword matching)
# ---------------------------------------------------------------------------

def extract_skills(text: str, skills: list[dict]) -> list[str]:
    """Return the names of the skills whose keywords appear in the text.
    Matching is case-insensitive and word-boundary aware, so 'git' does
    not match 'github' and 'ml' does not match 'html'."""
    lowered = text.lower()
    found = []
    for skill in skills:
        for keyword in skill["keywords"]:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, lowered):
                found.append(skill["name"])
                break
    return found


# ---------------------------------------------------------------------------
# Embeddings: MiniLM first, TF-IDF fallback
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors, clipped to [0, 1].
    Returns 0.0 if either vector is all zeros."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, score))


class MiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2, 384-dim embeddings."""

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class TfidfEmbedder:
    """TF-IDF fallback. Fit on the corpus (resume + JDs) once, then
    transform. Not semantic -- it only counts shared words -- but it
    keeps the tool working with no model download."""

    name = "TF-IDF (scikit-learn fallback)"

    def __init__(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(float)


def get_embedder(corpus: list[str]):
    """Try MiniLM; on any failure fall back to TF-IDF.
    Returns (embedder, method_name) and prints which path was taken."""
    try:
        embedder = MiniLMEmbedder()
        print(f"Embeddings: {embedder.name}")
        return embedder, "minilm"
    except Exception as exc:  # noqa: BLE001 - any load failure triggers fallback
        print(f"MiniLM unavailable ({exc}). Falling back to TF-IDF.")
        return TfidfEmbedder(corpus), "tfidf"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_jobs(
    resume_text: str,
    jobs: list[Job],
    skills: list[dict],
    embedder=None,
) -> list[Job]:
    """Score every job against the resume and attach skill-gap info.
    Returns the jobs sorted by score, highest first."""
    if not jobs:
        return []
    if embedder is None:
        embedder, _ = get_embedder([resume_text] + [j.text for j in jobs])

    vectors = embedder.encode([resume_text] + [j.text for j in jobs])
    resume_vec = vectors[0]
    resume_skills = set(extract_skills(resume_text, skills))

    for i, job in enumerate(jobs):
        job.score = cosine_similarity(resume_vec, vectors[i + 1])
        required = extract_skills(job.required_text or job.text, skills)
        job.required_skills = required
        job.matched_skills = [s for s in required if s in resume_skills]
        job.missing_skills = [s for s in required if s not in resume_skills]

    return sorted(jobs, key=lambda j: j.score, reverse=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_table(ranked: list[Job]) -> str:
    lines = [
        f"{'Rank':<5}{'Role':<42}{'Company':<18}{'Score':<7}Top missing skills",
        "-" * 110,
    ]
    for i, job in enumerate(ranked, start=1):
        missing = ", ".join(job.missing_skills[:3]) or "none"
        lines.append(
            f"{i:<5}{job.title[:40]:<42}{job.company[:16]:<18}"
            f"{job.score * 100:>5.1f}  {missing}"
        )
    return "\n".join(lines)


def suggestion_for(job: Job, rank: int) -> str:
    """One-paragraph suggestion per job based on its gaps."""
    if not job.missing_skills:
        return (
            f"#{rank} {job.title} at {job.company}: strongest fit — the resume "
            "already covers every required skill listed in this JD. Apply first "
            "and spend the application time tailoring the resume wording to the "
            "JD instead of learning anything new."
        )
    top = ", ".join(job.missing_skills[:3])
    rest = f" plus {len(job.missing_skills) - 3} more" if len(job.missing_skills) > 3 else ""
    return (
        f"#{rank} {job.title} at {job.company}: a good fit with a clear gap. "
        f"The resume is missing {top}{rest}. The fastest win is to close the "
        f"top gap with one small project or a short course and add it to the "
        "resume before applying — a single project that uses the missing skill "
        "is worth more than listing the keyword alone."
    )


def build_report(ranked: list[Job], method: str) -> str:
    lines = [
        "# JD Match Report",
        "",
        f"Embedding method: {method}",
        "",
        "## Ranked jobs",
        "",
        "| Rank | Role | Company | Score | Top missing skills |",
        "|------|------|---------|-------|--------------------|",
    ]
    for i, job in enumerate(ranked, start=1):
        missing = ", ".join(job.missing_skills[:3]) or "none"
        lines.append(
            f"| {i} | {job.title} | {job.company} | {job.score * 100:.1f} | {missing} |"
        )
    lines += ["", "## Suggestions", ""]
    for i, job in enumerate(ranked, start=1):
        lines.append(suggestion_for(job, i))
        lines.append("")
    return "\n".join(lines)


def cmd_match(args) -> int:
    resume_text = read_text(args.resume)
    skills = load_skills(args.skills)
    jobs = load_jobs(args.jds)
    if not jobs:
        print(f"No job descriptions found in {args.jds}. Add .md files and retry.")
        return 0
    embedder, method = get_embedder([resume_text] + [j.text for j in jobs])
    ranked = rank_jobs(resume_text, jobs, skills, embedder=embedder)
    print()
    print(format_table(ranked))
    print()
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(build_report(ranked, method))
    print(f"Report written to {args.report}")
    return 0


def cmd_gaps(args) -> int:
    resume_text = read_text(args.resume)
    skills = load_skills(args.skills)
    jobs = load_jobs(args.jds)
    if not jobs:
        print(f"No job descriptions found in {args.jds}. Add .md files and retry.")
        return 0
    embedder, _ = get_embedder([resume_text] + [j.text for j in jobs])
    ranked = rank_jobs(resume_text, jobs, skills, embedder=embedder)
    if args.job < 1 or args.job > len(ranked):
        print(f"Job #{args.job} does not exist. There are {len(ranked)} ranked jobs.")
        return 1
    job = ranked[args.job - 1]
    print()
    print(f"#{args.job} {job.title} — {job.company} ({job.location})")
    print(f"Match score: {job.score * 100:.1f}/100")
    print(f"Required skills ({len(job.required_skills)}): {', '.join(job.required_skills)}")
    print(f"Matched skills ({len(job.matched_skills)}): {', '.join(job.matched_skills) or 'none'}")
    print(f"Missing skills ({len(job.missing_skills)}): {', '.join(job.missing_skills) or 'none'}")
    print()
    print(suggestion_for(job, args.job))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank job descriptions against a resume and explain skill gaps."
    )
    parser.add_argument("--resume", default=DEFAULT_RESUME, help="Path to resume .md file")
    parser.add_argument("--jds", default=DEFAULT_JDS_DIR, help="Folder of JD .md files")
    parser.add_argument("--skills", default=DEFAULT_SKILLS, help="Path to skills.yaml")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Where to write report.md")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("match", help="Rank all JDs and write the report")
    gaps = sub.add_parser("gaps", help="Show skill-gap detail for one ranked job")
    gaps.add_argument("--job", type=int, required=True, help="Rank number from the match table")
    args = parser.parse_args(argv)
    if args.command == "match":
        return cmd_match(args)
    return cmd_gaps(args)


if __name__ == "__main__":
    sys.exit(main())
