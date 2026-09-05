"""A local model registry.

Models were a file on disk plus a `metadata.json`. Training overwrote both. That has two
consequences worth naming: the model that produced last week's numbers no longer exists,
so a result cannot be reproduced; and there is no notion of a model being *approved* --
whatever was trained last is what serves, including an experiment someone ran on a laptop.

The registry fixes both by making a version an immutable directory and a status a
deliberate act:

    EXPERIMENTAL -> STAGING -> PRODUCTION -> RETIRED

`promote()` is the only path to PRODUCTION and it checks the model against a quality
floor first, so promoting a model that is worse than the one it replaces requires
explicitly saying so. `load_production()` returns the approved model and refuses to
silently fall back to a newer unapproved one -- a registry that serves whatever is latest
is a directory with extra steps.

Deliberately not MLflow. The requirement is "an immutable directory per version plus a
JSON index", the dependency is a server and a database, and the seam is such that swapping
in a real registry later means implementing four methods.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from backend.app.config import MODEL_DIR


class ModelStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


#: Only one model may be PRODUCTION at a time. Promoting retires the incumbent, which is
#: what makes "which model served this decision?" answerable from the index alone.
LEGAL_TRANSITIONS: dict[ModelStatus, frozenset[ModelStatus]] = {
    ModelStatus.EXPERIMENTAL: frozenset({ModelStatus.STAGING, ModelStatus.RETIRED}),
    ModelStatus.STAGING: frozenset({ModelStatus.PRODUCTION, ModelStatus.RETIRED,
                                    ModelStatus.EXPERIMENTAL}),
    ModelStatus.PRODUCTION: frozenset({ModelStatus.RETIRED}),
    #: A retired model may be reinstated to staging -- rollback is a real operation and a
    #: registry that cannot roll back is a registry people work around.
    ModelStatus.RETIRED: frozenset({ModelStatus.STAGING}),
}


class RegistryError(RuntimeError):
    """Any refusal: an illegal transition, a failed quality gate, a missing version."""


@dataclass
class ModelRecord:
    """One immutable version."""
    version: str
    name: str
    status: str = ModelStatus.EXPERIMENTAL.value
    algorithm: str = ""
    created_at: str = ""
    #: Hash of the training data, so "which rows produced this model" is answerable.
    training_data_hash: str = ""
    training_dataset_version: str = ""
    #: Hash of the feature list, so a feature-schema change cannot silently be served
    #: against a model trained on the old one.
    feature_schema_hash: str = ""
    features: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artefacts: list[str] = field(default_factory=list)
    seed: int | None = None
    git_commit: str = ""
    notes: str = ""
    promoted_at: str = ""
    promoted_by: str = ""
    retired_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#: The floor a model must clear to reach PRODUCTION. Not a formality: without it the
#: registry records a lifecycle it does not enforce, which is the failure mode of most
#: model governance.
@dataclass(frozen=True)
class QualityGate:
    min_roc_auc: float = 0.60
    max_brier: float = 0.25
    #: A candidate may not be materially worse than the incumbent on ranking. Small
    #: regressions are allowed -- a model can trade a little AUC for better calibration --
    #: but a cliff requires an explicit override.
    max_roc_auc_regression: float = 0.02

    def check(self, candidate: ModelRecord,
              incumbent: ModelRecord | None) -> list[str]:
        problems: list[str] = []
        auc = float(candidate.metrics.get("roc_auc", 0.0))
        brier = float(candidate.metrics.get("brier", 1.0))
        if auc < self.min_roc_auc:
            problems.append(f"ROC-AUC {auc:.4f} is below the {self.min_roc_auc} floor")
        if brier > self.max_brier:
            problems.append(f"Brier {brier:.4f} exceeds the {self.max_brier} ceiling")
        if incumbent is not None:
            prev = float(incumbent.metrics.get("roc_auc", 0.0))
            if prev - auc > self.max_roc_auc_regression:
                problems.append(
                    f"ROC-AUC regresses {prev - auc:.4f} against the incumbent "
                    f"{incumbent.version} ({prev:.4f} -> {auc:.4f}), beyond the "
                    f"{self.max_roc_auc_regression} tolerance")
        return problems


class ModelRegistry:
    """Filesystem-backed. One directory per version, one JSON index."""

    INDEX = "registry.json"

    def __init__(self, root: Path | str = MODEL_DIR):
        self.root = Path(root) / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / self.INDEX

    # ------------------------------------------------------------------ index
    def _read(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text())

    def _write(self, index: dict[str, dict]) -> None:
        # Write-then-rename: a crash mid-write must not leave a truncated index that
        # loses every registered model.
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, default=str))
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------ register
    def register(self, name: str, artefact_dir: Path | str, *,
                 metrics: dict[str, float] | None = None,
                 features: list[str] | None = None,
                 algorithm: str = "", params: dict | None = None,
                 training_data_hash: str = "", training_dataset_version: str = "",
                 seed: int | None = None, git_commit: str = "",
                 notes: str = "", version: str | None = None) -> ModelRecord:
        """Copy artefacts into an immutable version directory and index them."""
        src = Path(artefact_dir)
        if not src.exists():
            raise RegistryError(f"no artefacts at {src}")

        ver = version or self._next_version(name)
        dest = self.root / ver
        if dest.exists():
            raise RegistryError(
                f"version {ver} already exists. Versions are immutable: overwriting one "
                f"would break every stored prediction that names it.")
        dest.mkdir(parents=True)

        copied: list[str] = []
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix in (".json", ".joblib", ".ubj", ".txt"):
                shutil.copy2(f, dest / f.name)
                copied.append(f.name)

        feats = features or []
        record = ModelRecord(
            version=ver, name=name, algorithm=algorithm,
            created_at=datetime.now(UTC).isoformat(),
            training_data_hash=training_data_hash,
            training_dataset_version=training_dataset_version,
            feature_schema_hash=self.feature_schema_hash(feats),
            features=feats, metrics=metrics or {}, params=params or {},
            artefacts=copied, seed=seed, git_commit=git_commit, notes=notes,
        )
        index = self._read()
        index[ver] = record.to_dict()
        self._write(index)
        (dest / "record.json").write_text(json.dumps(record.to_dict(), indent=2, default=str))
        return record

    def _next_version(self, name: str) -> str:
        existing = [v for v in self._read() if v.startswith(f"{name}-v")]
        n = 0
        for v in existing:
            try:
                n = max(n, int(v.rsplit("-v", 1)[1]))
            except (IndexError, ValueError):
                continue
        return f"{name}-v{n + 1}"

    @staticmethod
    def feature_schema_hash(features: list[str]) -> str:
        """Order-sensitive: a reordered feature matrix is a different matrix, and a model
        served against one would produce plausible nonsense rather than an error."""
        return hashlib.sha256("|".join(features).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ lifecycle
    def transition(self, version: str, to: ModelStatus, *, actor: str = "system",
                   gate: QualityGate | None = None, force: bool = False,
                   reason: str = "") -> ModelRecord:
        index = self._read()
        if version not in index:
            raise RegistryError(f"unknown model version {version!r}")
        record = ModelRecord(**index[version])
        frm = ModelStatus(record.status)

        if to not in LEGAL_TRANSITIONS[frm]:
            raise RegistryError(
                f"{version}: {frm.value} -> {to.value} is not a legal transition "
                f"(allowed: {sorted(s.value for s in LEGAL_TRANSITIONS[frm])})")

        if to is ModelStatus.PRODUCTION:
            incumbent = self.production()
            problems = (gate or QualityGate()).check(record, incumbent)
            if problems and not force:
                raise RegistryError(
                    f"{version} failed the promotion gate: " + "; ".join(problems)
                    + ". Pass force=True with a reason to promote anyway -- and the "
                      "reason is recorded.")
            if incumbent is not None and incumbent.version != version:
                index[incumbent.version]["status"] = ModelStatus.RETIRED.value
                index[incumbent.version]["retired_at"] = \
                    datetime.now(UTC).isoformat()
            record.promoted_at = datetime.now(UTC).isoformat()
            record.promoted_by = actor
            if problems and force:
                record.notes = (record.notes + " | FORCED PROMOTION despite: "
                                + "; ".join(problems) + f" | reason: {reason}").strip(" |")

        if to is ModelStatus.RETIRED:
            record.retired_at = datetime.now(UTC).isoformat()

        record.status = to.value
        index[version] = record.to_dict()
        self._write(index)
        return record

    def promote(self, version: str, actor: str = "system", **kw) -> ModelRecord:
        return self.transition(version, ModelStatus.PRODUCTION, actor=actor, **kw)

    # ------------------------------------------------------------------ read
    def get(self, version: str) -> ModelRecord | None:
        raw = self._read().get(version)
        return ModelRecord(**raw) if raw else None

    def list(self, status: ModelStatus | None = None) -> list[ModelRecord]:
        records = [ModelRecord(**r) for r in self._read().values()]
        if status is not None:
            records = [r for r in records if r.status == status.value]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def production(self) -> ModelRecord | None:
        live = self.list(ModelStatus.PRODUCTION)
        if len(live) > 1:
            raise RegistryError(
                f"{len(live)} models are marked PRODUCTION ({[r.version for r in live]}). "
                f"The registry cannot say which one served a decision, so it refuses to "
                f"guess.")
        return live[0] if live else None

    def artefact_dir(self, version: str) -> Path:
        d = self.root / version
        if not d.exists():
            raise RegistryError(f"no artefacts for {version}")
        return d

    def load_production(self, strict: bool = True) -> tuple[ModelRecord, Path] | None:
        """The approved model and where its files are.

        `strict=True` (the default) returns None when nothing is approved, rather than
        falling back to the newest experimental model. That refusal is the point: a
        production process that quietly serves an unapproved model has a registry for
        decoration.
        """
        record = self.production()
        if record is None:
            if strict:
                return None
            candidates = self.list(ModelStatus.STAGING) or self.list()
            if not candidates:
                return None
            record = candidates[0]
        return record, self.artefact_dir(record.version)

    def describe(self) -> dict:
        records = self.list()
        prod = next((r for r in records if r.status == ModelStatus.PRODUCTION.value), None)
        return {
            "root": str(self.root),
            "versions": len(records),
            "production": prod.version if prod else None,
            "by_status": {s.value: sum(1 for r in records if r.status == s.value)
                          for s in ModelStatus},
            "models": [{"version": r.version, "status": r.status,
                        "created_at": r.created_at, "roc_auc": r.metrics.get("roc_auc"),
                        "brier": r.metrics.get("brier")} for r in records],
        }
