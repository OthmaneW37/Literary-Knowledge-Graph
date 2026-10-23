"""Open any catalog book in one click: download, index, then embed in the background.

The book becomes usable for reading, keyword search and questions as soon as
it is indexed (a few seconds). Vector embeddings, which can take a minute for
a long novel, are computed afterwards; meanwhile the retriever uses BM25.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from .gutendex import CatalogError

STAGES = {
    "queued": "En attente",
    "downloading": "Téléchargement du texte",
    "indexing": "Lecture et découpage du livre",
    "ready": "Prêt — recherche sémantique en préparation",
    "done": "Prêt",
    "failed": "Échec",
}


@dataclass
class OpenJob:
    job_id: str
    provider_id: int | str
    provider: str = "gutenberg"
    title: str = ""
    state: str = "queued"
    stage: str = STAGES["queued"]
    work_id: str | None = None
    already_installed: bool = False
    embedded: int = 0
    to_embed: int = 0
    error: str | None = None
    warning: str | None = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class BookOpener:
    def __init__(self, catalog, library, reload_assistant, max_jobs: int = 50, wikisource=None) -> None:
        """`reload_assistant()` rebuilds the index and returns the new assistant."""
        self.catalog = catalog
        self.wikisource = wikisource
        self.library = library
        self.reload_assistant = reload_assistant
        self.max_jobs = max_jobs
        self.jobs: dict[str, OpenJob] = {}
        self._install_lock = threading.Lock()
        self._lock = threading.Lock()

    def open(self, provider_id: int | str, provider: str = "gutenberg") -> OpenJob:
        if provider == "wikisource" and self.wikisource is None:
            raise ValueError("Wikisource n’est pas configuré.")
        with self._lock:
            active = next((job for job in self.jobs.values() if job.provider == provider
                           and job.provider_id == provider_id and job.state in {"queued", "downloading", "indexing"}), None)
            if active:
                return active
            job = OpenJob(job_id=uuid4().hex, provider_id=provider_id, provider=provider)
            self.jobs[job.job_id] = job
            for old in sorted(self.jobs.values(), key=lambda item: item.started_at)[:-self.max_jobs]:
                self.jobs.pop(old.job_id, None)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> OpenJob | None:
        return self.jobs.get(job_id)

    def _set(self, job: OpenJob, state: str, **values) -> None:
        job.state, job.stage = state, STAGES[state]
        for key, value in values.items():
            setattr(job, key, value)

    def _run(self, job: OpenJob) -> None:
        try:
            if job.provider == "wikisource":
                self._install_wikisource(job)
            else:
                self._install_gutenberg(job)
            self._embed(job)
        except (CatalogError, ValueError, OSError) as exc:
            self._set(job, "failed", error=str(exc) or "Ouverture impossible.")
        except Exception as exc:  # Never leave a job silently stuck.
            logging.exception("Opening book %s failed", job.provider_id)
            self._set(job, "failed", error=f"Ouverture impossible : {exc.__class__.__name__}.")

    def _install_gutenberg(self, job: OpenJob) -> None:
        book = self.catalog.get_book(int(job.provider_id))
        job.title = book.title
        if book.work_id in self.library.installed_ids():
            job.work_id, job.already_installed = book.work_id, True
            return
        self._set(job, "downloading")
        downloaded = self.catalog.download(book)
        self._set(job, "indexing")
        # Indexing rewrites shared library files; one book at a time.
        with self._install_lock:
            installed = self.library.install(book, downloaded)
        job.work_id, job.already_installed = installed.work_id, installed.already_installed

    def _install_wikisource(self, job: OpenJob) -> None:
        from .wikisource import work_id_for
        language, _, title = str(job.provider_id).partition(":")
        job.title, work_id = title, work_id_for(title, language)
        if work_id in self.library.installed_ids():
            job.work_id, job.already_installed = work_id, True
            return
        self._set(job, "downloading")
        book = self.wikisource.fetch(str(job.provider_id))
        self._set(job, "indexing")
        with self._install_lock:
            installed = self.library.install_wikisource(work_id, book)
        job.work_id, job.already_installed = installed.work_id, installed.already_installed

    def _embed(self, job: OpenJob) -> None:
        assistant = self.reload_assistant()
        semantic = getattr(assistant.hybrid_retriever, "semantic", None)
        if semantic is None or semantic.ready:
            self._set(job, "done")
            return
        self._set(job, "ready")
        try:
            semantic.prepare(progress=lambda done, total: self._progress(job, done, total))
        except Exception as exc:
            logging.warning("Background embeddings failed for %s: %s", job.work_id, exc)
            job.warning = "Livre prêt pour la recherche par mots-clés ; le service d’embeddings est indisponible."
        self._set(job, "done")

    @staticmethod
    def _progress(job: OpenJob, done: int, total: int) -> None:
        job.embedded, job.to_embed = done, total
