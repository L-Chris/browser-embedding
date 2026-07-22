"""Application service that orchestrates one complete training run."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from browser_embedding.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from browser_embedding.config import ExperimentConfig, resolve_path
from browser_embedding.data import build_data
from browser_embedding.evaluation import evaluate_multilingual_seed, evaluate_pairs
from browser_embedding.export import export_model
from browser_embedding.model import BrowserEncoder
from browser_embedding.objectives import MatryoshkaObjective
from browser_embedding.quantization import set_quantization_strength, ternary_health


def select_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested!r}, but CUDA is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("requested 'mps', but MPS is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def nested_metric(metrics: dict[str, Any], path: str) -> float:
    value: Any = metrics
    for segment in path.split("."):
        value = value[segment]
    if not isinstance(value, (float, int)):
        raise TypeError(f"metric {path!r} is not numeric")
    return float(value)


class TrainingRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        project_root: Path,
        *,
        resume: Path | None = None,
        max_batches: int | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.device = select_device(config.training.device)
        self.run_dir = resolve_path(project_root, config.run.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.resume_path = resume
        self.max_batches = max_batches

        seed_everything(config.seed)
        self.data = build_data(
            config.data,
            config.model,
            config.training,
            project_root=project_root,
            seed=config.seed,
            device=self.device,
        )
        self.model = BrowserEncoder(config.model).to(self.device)
        self.objective = MatryoshkaObjective(config.model, config.objective)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fused=self.device.type == "cuda",
        )
        batches = min(len(self.data.train_loader), max_batches or len(self.data.train_loader))
        total_steps = batches * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: learning_rate_multiplier(step, total_steps, warmup_steps),
        )
        self.start_epoch = 1
        self.global_step = 0
        self.best_value = -math.inf
        if resume is not None:
            self._restore(resume)

    def _restore(self, path: Path) -> None:
        checkpoint = load_checkpoint(path)
        if checkpoint["experiment"] != self.config.model_dump(mode="json"):
            raise ValueError("resume checkpoint experiment does not match the config")
        self.model.load_state_dict(checkpoint["state"]["model"])
        self.optimizer.load_state_dict(checkpoint["state"]["optimizer"])
        self.scheduler.load_state_dict(  # type: ignore[no-untyped-call]
            checkpoint["state"]["scheduler"]
        )
        restore_rng_state(checkpoint["rng"], self.data.generator)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_value = float(checkpoint["best_value"])

    def _write_manifest(self) -> None:
        estimate = self.config.deployment.estimate(self.config.model)
        manifest = {
            "schema_version": 1,
            "started_at": datetime.now(UTC).isoformat(),
            "experiment": self.config.model_dump(mode="json"),
            "device": str(self.device),
            "torch_version": torch.__version__,
            "model_parameters": self.model.parameter_counts(),
            "deployment_estimate": estimate.as_dict(),
            "data": self.data.provenance,
            "resume": str(self.resume_path) if self.resume_path else None,
            "max_batches": self.max_batches,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _checkpoint_payload(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_value": self.best_value,
            "experiment": self.config.model_dump(mode="json"),
            "state": {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),  # type: ignore[no-untyped-call]
            },
            "metrics": metrics,
            "data_provenance": self.data.provenance,
            "rng": capture_rng_state(self.data.generator),
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        strength = (
            1.0
            if self.config.quantization.enabled and epoch > self.config.quantization.warmup_epochs
            else 0.0
        )
        set_quantization_strength(self.model, strength)
        self.model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        for batch_index, batch in enumerate(self.data.train_loader):
            if self.max_batches is not None and batch_index >= self.max_batches:
                break
            token_ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            teacher = batch["teacher_embeddings"].to(self.device, non_blocking=True)
            pairs, roles, sequence = token_ids.shape
            output = self.model(
                token_ids.reshape(pairs * roles, sequence), mask.reshape(pairs * roles, sequence)
            ).reshape(pairs, roles, -1)
            result = self.objective(output, teacher, batch["group_id"])
            self.optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            steps += 1
            for name, value in result.components.items():
                totals[name] += value
            totals["gradient_norm"] += float(gradient_norm)
            if self.global_step % self.config.training.log_every == 0:
                print(
                    f"epoch={epoch} step={self.global_step} loss={result.components['total']:.4f} "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        if steps == 0:
            raise RuntimeError("training loader produced no batches")
        metrics = {name: value / steps for name, value in totals.items()}
        metrics["quantization_strength"] = strength
        return metrics

    def run(self) -> Path:
        if self.start_epoch > self.config.training.epochs:
            raise ValueError("checkpoint is already past the requested final epoch")
        self._write_manifest()
        history_path = self.run_dir / "metrics.jsonl"
        history_mode = "a" if self.resume_path else "w"
        with history_path.open(history_mode, encoding="utf-8") as history:
            for epoch in range(self.start_epoch, self.config.training.epochs + 1):
                started = time.perf_counter()
                train_metrics = self._train_epoch(epoch)
                validation = evaluate_pairs(self.model, self.data.validation_loader, self.device)
                metrics: dict[str, Any] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "train": train_metrics,
                    "validation": validation,
                }
                if self.config.evaluation.multilingual_seed_enabled:
                    tokenizer_path = self.config.data.tokenizer_path
                    seed_path = self.config.evaluation.seed_path
                    assert tokenizer_path is not None and seed_path is not None
                    metrics["multilingual"] = evaluate_multilingual_seed(
                        self.model,
                        tokenizer_path=resolve_path(self.project_root, tokenizer_path),
                        seed_path=resolve_path(self.project_root, seed_path),
                        max_sequence_length=self.config.model.max_sequence_length,
                        batch_size=self.config.training.eval_batch_size,
                        device=self.device,
                    )
                health = ternary_health(self.model)
                metrics["quantization"] = {
                    "zero_fraction_mean": float(np.mean(list(health.values()))),
                    "zero_fraction_by_layer": health,
                }
                score = nested_metric(metrics, self.config.evaluation.best_metric)
                is_best = score > self.best_value
                self.best_value = max(self.best_value, score)
                history.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                history.flush()
                payload = self._checkpoint_payload(epoch, metrics)
                if epoch % self.config.training.save_every == 0:
                    save_checkpoint(self.run_dir / f"checkpoint-{epoch:03d}.pt", payload)
                save_checkpoint(self.run_dir / "last.pt", payload)
                if is_best:
                    save_checkpoint(self.run_dir / "best.pt", payload)
                print(
                    f"epoch={epoch} validation_ndcg={validation['ndcg_at_10']:.4f} "
                    f"best={self.best_value:.4f}",
                    flush=True,
                )

        best_path = self.run_dir / "best.pt"
        if self.config.run.export_on_finish:
            best = load_checkpoint(best_path)
            self.model.load_state_dict(best["state"]["model"])
            manifest = export_model(self.model, self.config, self.run_dir / "model.bem")
            print(f"exported={self.run_dir / 'model.bem'} bytes={manifest['bytes']}", flush=True)
        return best_path
