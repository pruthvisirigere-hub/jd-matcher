"""Tests for jd-matcher. They use the TF-IDF embedder (fast, no model
download) except for the fallback-path test, which only checks that the
TF-IDF class itself works end to end."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402

SKILLS = [
    {"name": "Python", "keywords": ["python"]},
    {"name": "PyTorch", "keywords": ["pytorch", "torch"]},
    {"name": "Kubernetes", "keywords": ["kubernetes", "k8s"]},
    {"name": "SQL", "keywords": ["sql"]},
    {"name": "Git", "keywords": ["git"]},
]

RESUME = (
    "Python developer with machine learning experience. "
    "Built models with PyTorch and scikit-learn. Writes SQL queries daily."
)


def make_job(title, text):
    return main.Job(
        title=title, company="TestCo", location="Remote", text=text, filename="t.md"
    )


def test_ranking_puts_most_relevant_jd_first():
    """Controlled input: one JD is about Python/PyTorch (matches the
    resume), the other is about Kubernetes (does not)."""
    jobs = [
        make_job("Kubernetes Engineer", "We need Kubernetes, k8s, containers, Helm, DevOps."),
        make_job("Python ML Engineer", "Python, PyTorch, machine learning, SQL, models."),
    ]
    embedder = main.TfidfEmbedder([RESUME] + [j.text for j in jobs])
    ranked = main.rank_jobs(RESUME, jobs, SKILLS, embedder=embedder)
    assert ranked[0].title == "Python ML Engineer"
    assert ranked[0].score > ranked[1].score


def test_gap_detection_finds_known_missing_skill():
    """The JD requires Kubernetes, the resume never mentions it."""
    jobs = [make_job("Platform Engineer", "Python and Kubernetes and SQL required.")]
    embedder = main.TfidfEmbedder([RESUME] + [j.text for j in jobs])
    ranked = main.rank_jobs(RESUME, jobs, SKILLS, embedder=embedder)
    job = ranked[0]
    assert "Kubernetes" in job.missing_skills
    assert "Python" in job.matched_skills
    assert "SQL" in job.matched_skills


def test_empty_jd_folder_handled_gracefully(tmp_path):
    """An empty (or missing) JD folder returns no jobs and rank_jobs
    returns an empty list instead of crashing."""
    assert main.load_jobs(str(tmp_path)) == []
    assert main.rank_jobs(RESUME, [], SKILLS) == []


def test_report_file_is_created(tmp_path):
    """Running the match command writes results/report.md with a table."""
    report = tmp_path / "report.md"
    argv = [
        "--resume", os.path.join(os.path.dirname(__file__), "..", "data", "sample_resume.md"),
        "--jds", os.path.join(os.path.dirname(__file__), "..", "data", "jds"),
        "--report", str(report),
        "match",
    ]
    rc = main.main(argv)
    assert rc == 0
    assert report.exists()
    text = report.read_text()
    assert "JD Match Report" in text
    assert "| Rank |" in text


def test_cosine_similarity_is_in_zero_to_one():
    import numpy as np

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    score = main.cosine_similarity(a, b)
    assert 0.0 <= score <= 1.0
    # Identical vectors score 1.0; opposite vectors score 0.0 (clipped).
    assert main.cosine_similarity(a, a) == pytest.approx(1.0)
    assert main.cosine_similarity(a, -a) == pytest.approx(0.0)
    # Zero vector is handled, not a crash.
    assert main.cosine_similarity(a, np.zeros(3)) == 0.0


def test_tfidf_fallback_path_works():
    """The TF-IDF embedder produces real vectors and sane similarities."""
    embedder = main.TfidfEmbedder([RESUME, "Kubernetes containers DevOps"])
    vecs = embedder.encode([RESUME, "Python PyTorch machine learning"])
    assert vecs.shape[0] == 2
    similar = main.cosine_similarity(vecs[0], vecs[1])
    assert 0.0 < similar <= 1.0


def test_skill_extraction_is_word_boundary_aware():
    """'git' must not match inside 'github'."""
    assert main.extract_skills("See my work on github.com/alex", SKILLS) == []
    assert "Git" in main.extract_skills("I use Git every day.", SKILLS)


def test_nice_to_have_skills_are_not_counted_as_missing(tmp_path):
    """Only the 'Required Skills' section counts. A skill listed under
    'Nice to Have' must not appear in required or missing skills."""
    jd_file = tmp_path / "jd.md"
    jd_file.write_text(
        "# T\nCompany: TestCo\nLocation: Remote\n\n"
        "## Required Skills\n- Python\n\n## Nice to Have\n- Kubernetes\n"
    )
    job = main.parse_jd(str(jd_file))
    embedder = main.TfidfEmbedder([RESUME, job.text])
    ranked = main.rank_jobs(RESUME, [job], SKILLS, embedder=embedder)
    assert ranked[0].required_skills == ["Python"]
    assert "Kubernetes" not in ranked[0].missing_skills
