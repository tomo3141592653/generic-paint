#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# For optional display
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    _HAS_DISPLAY = True
except ImportError:
    _HAS_DISPLAY = False

# --- Optional heavy deps (loaded lazily) ---
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# =====================
# Genome & Rendering
# =====================

SHAPE_TYPES = ("RECT", "ELLIPSE")


@dataclass
class Gene:
    shape_type: str
    x: float
    y: float
    w: float
    h: float
    z: float
    r: int
    g: int
    b: int
    a: int

    @staticmethod
    def random(rng: random.Random) -> "Gene":
        w = max(0.02, rng.random() ** 2) * 0.9
        h = max(0.02, rng.random() ** 2) * 0.9
        return Gene(
            shape_type=rng.choice(SHAPE_TYPES),
            x=rng.random(),
            y=rng.random(),
            w=w,
            h=h,
            z=rng.random(),
            r=rng.randint(0, 255),
            g=rng.randint(0, 255),
            b=rng.randint(0, 255),
            a=rng.randint(30, 200),
        )


@dataclass
class DNA:
    genes: List[Gene]

    @staticmethod
    def random(gene_count: int, rng: random.Random) -> "DNA":
        return DNA(genes=[Gene.random(rng) for _ in range(gene_count)])


class Renderer:
    def __init__(self, width: int = 256, height: int = 256, verbose: bool = False):
        self.size = (width, height)
        self.verbose = verbose
        self.timing_stats = {
            "render_total": 0.0,
            "render_count": 0,
            "batch_render_total": 0.0,
            "batch_render_count": 0,
        }

    def render(self, dna: DNA) -> Image.Image:
        start_time = time.time()
        img = Image.new("RGBA", self.size, (0, 0, 0, 0))
        for g in sorted(dna.genes, key=lambda x: x.z):
            self._draw_gene(img, g)

        end_time = time.time()
        render_time = end_time - start_time
        self.timing_stats["render_total"] += render_time
        self.timing_stats["render_count"] += 1

        if self.verbose:
            print(f"Single render time: {render_time:.4f}s (genes: {len(dna.genes)})")

        return img

    def render_batch(self, dnas: List[DNA], n_workers: int = 4) -> List[Image.Image]:
        """Render multiple DNAs in parallel"""
        start_time = time.time()

        if n_workers == 1:
            results = [self.render(dna) for dna in dnas]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(self.render, dnas))

        end_time = time.time()
        batch_time = end_time - start_time
        self.timing_stats["batch_render_total"] += batch_time
        self.timing_stats["batch_render_count"] += 1

        if self.verbose:
            print(
                f"Batch render time: {batch_time:.4f}s ({len(dnas)} images, {n_workers} workers)"
            )
            print(f"  Avg per image: {batch_time/len(dnas):.4f}s")

        return results

    def get_timing_stats(self):
        """Get performance statistics"""
        stats = self.timing_stats.copy()
        if stats["render_count"] > 0:
            stats["avg_render_time"] = stats["render_total"] / stats["render_count"]
        if stats["batch_render_count"] > 0:
            stats["avg_batch_time"] = (
                stats["batch_render_total"] / stats["batch_render_count"]
            )
        return stats

    @staticmethod
    def _draw_gene(img: Image.Image, gene: Gene):
        W, H = img.size
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        cx = int(gene.x * W)
        cy = int(gene.y * H)
        ww = max(1, int(gene.w * W))
        hh = max(1, int(gene.h * H))
        x0 = cx - ww // 2
        y0 = cy - hh // 2
        x1 = x0 + ww
        y1 = y0 + hh
        color = (gene.r, gene.g, gene.b, gene.a)
        if gene.shape_type == "RECT":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)
        img.alpha_composite(layer)


# =====================
# CLIP Scorer
# =====================


class CLIPScorer:
    """
    Scores PIL images vs text prompts using CLIP cosine similarity.
    Aggregation across prompts: mean or max.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        prompt_agg: str = "mean",
        normalize: bool = True,
        use_fp16: bool = True,
        use_sdpa: bool = True,
        verbose: bool = False,
    ):
        if not _HAS_CLIP:
            raise RuntimeError(
                "transformers/torch not available. Install deps to use CLIPScorer."
            )

        # Store optimization settings
        self.use_fp16 = use_fp16
        self.use_sdpa = use_sdpa
        self.verbose = verbose
        self.timing_stats = {
            "set_prompts_total": 0.0,
            "set_prompts_count": 0,
            "score_total": 0.0,
            "score_count": 0,
            "score_batch_total": 0.0,
            "score_batch_count": 0,
        }

        # Auto-detect best device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            # Check CUDA capability
            import subprocess

            try:
                # Try to use CUDA if available and compatible
                self.device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except:
                self.device = "cpu"
                print("GPU detected but not compatible, using CPU")
        else:
            self.device = "cpu"

        # Set up model loading parameters
        model_kwargs = {}
        if self.use_fp16 and self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        # CLIP doesn't support SDPA yet, use eager attention
        model_kwargs["attn_implementation"] = "eager"

        self.model = CLIPModel.from_pretrained(model_name, **model_kwargs).to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.prompt_agg = prompt_agg
        self.normalize = normalize
        self._text_embeds = None

    def set_prompts(self, prompts: List[str]):
        start_time = time.time()

        if len(prompts) == 0:
            raise ValueError("At least one prompt is required.")
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        # Convert to fp16 if enabled
        if self.use_fp16 and self.device == "cuda":
            for key in inputs:
                if hasattr(inputs[key], "dtype") and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].half()

        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        if self.normalize:
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        self._text_embeds = text_embeds  # [P, D]

        end_time = time.time()
        prompt_time = end_time - start_time
        self.timing_stats["set_prompts_total"] += prompt_time
        self.timing_stats["set_prompts_count"] += 1

        if self.verbose:
            print(f"Set prompts time: {prompt_time:.4f}s ({len(prompts)} prompts)")

    def score(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        if self._text_embeds is None:
            raise RuntimeError("Call set_prompts() before scoring.")
        start_time = time.time()
        scores = []

        for i in range(0, len(images), batch_size):
            batch_start = time.time()
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            # Convert to fp16 if enabled
            if self.use_fp16 and self.device == "cuda":
                for key in inputs:
                    if (
                        hasattr(inputs[key], "dtype")
                        and inputs[key].dtype == torch.float32
                    ):
                        inputs[key] = inputs[key].half()

            with torch.no_grad():
                image_embeds = self.model.get_image_features(**inputs)
            if self.normalize:
                image_embeds = image_embeds / image_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
            # cosine sim with each prompt embedding, then aggregate
            sim = image_embeds @ self._text_embeds.T  # [B, P]
            if self.prompt_agg == "max":
                s = sim.max(dim=1).values
            else:
                s = sim.mean(dim=1)
            scores.extend(s.detach().cpu().numpy().tolist())

            batch_time = time.time() - batch_start
            self.timing_stats["score_batch_total"] += batch_time
            self.timing_stats["score_batch_count"] += 1

            if self.verbose:
                print(
                    f"  CLIP batch {i//batch_size + 1}: {batch_time:.4f}s ({len(batch)} images)"
                )

        end_time = time.time()
        total_time = end_time - start_time
        self.timing_stats["score_total"] += total_time
        self.timing_stats["score_count"] += 1

        if self.verbose:
            print(f"CLIP score total time: {total_time:.4f}s ({len(images)} images)")
            print(f"  Avg per image: {total_time/len(images):.4f}s")

        return np.array(scores, dtype=np.float32)

    def get_timing_stats(self):
        """Get performance statistics"""
        stats = self.timing_stats.copy()
        if stats["set_prompts_count"] > 0:
            stats["avg_set_prompts_time"] = (
                stats["set_prompts_total"] / stats["set_prompts_count"]
            )
        if stats["score_count"] > 0:
            stats["avg_score_time"] = stats["score_total"] / stats["score_count"]
        if stats["score_batch_count"] > 0:
            stats["avg_batch_time"] = (
                stats["score_batch_total"] / stats["score_batch_count"]
            )
        return stats


class DummyScorer:
    """Fallback random scorer (useful for dry runs without CLIP)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def set_prompts(self, prompts: List[str]):
        return

    def score(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        return np.array([self.rng.random() for _ in images], dtype=np.float32)


# =====================
# GA Operators
# =====================


@dataclass
class GAConfig:
    pop_size: int = 40
    genes_min: int = 3000
    genes_max: int = 3000
    elitism: int = 5  # 上位5個体が生存
    tournament_k: int = 4
    mutation_rate: float = 0.12  # per-gene param mutation prob
    add_gene_prob: float = 0.01  # 遺伝子数固定のため低く設定
    remove_gene_prob: float = 0.01  # 遺伝子数固定のため低く設定
    crossover: str = "elite_strategy"  # エリート優遇戦略
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "GAConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RenderConfig:
    width: int = 256
    height: int = 256

    @classmethod
    def from_dict(cls, data: dict) -> "RenderConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    generations: int = 30
    save_top_k: int = 6
    out_dir: str = "runs/demo"
    batch_size: int = 16
    render_workers: int = 8
    save_freq: int = 100
    show: bool = False
    show_top_k: int = 6

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    prompts: List[str] = field(
        default_factory=lambda: ["aesthetic curves", "smooth soft color gradients"]
    )
    model: str = "openai/clip-vit-base-patch32"
    dummy: bool = False
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    ga: GAConfig = field(default_factory=GAConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """設定ファイルから設定を読み込み"""
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML is required for config files. Install with: pip install pyyaml"
            )

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            ga=GAConfig.from_dict(data.get("ga", {})),
            render=RenderConfig.from_dict(data.get("render", {})),
            run=RunConfig.from_dict(data.get("run", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
        )

    def to_file(self, config_path: str):
        """設定をファイルに保存"""
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML is required for config files. Install with: pip install pyyaml"
            )

        data = {
            "ga": asdict(self.ga),
            "render": asdict(self.render),
            "run": asdict(self.run),
            "model": asdict(self.model),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )


class GeneticAlgorithm:
    def __init__(self, ga_cfg: GAConfig, rnd: random.Random):
        self.cfg = ga_cfg
        self.rng = rnd

    # ---- Selection ----
    def _tournament(self, population: List[Tuple[DNA, float]]) -> DNA:
        k = min(self.cfg.tournament_k, len(population))
        subset = self.rng.sample(population, k)
        subset.sort(key=lambda t: t[1], reverse=True)
        return subset[0][0]

    # ---- Crossover ----
    def _crossover(self, a: DNA, b: DNA) -> DNA:
        if self.cfg.crossover == "one_point":
            cut = self.rng.randint(1, min(len(a.genes), len(b.genes)) - 1)
            genes = a.genes[:cut] + b.genes[cut:]
        else:
            # uniform
            max_len = max(len(a.genes), len(b.genes))
            genes = []
            for i in range(max_len):
                pick_a = self.rng.random() < 0.5
                if pick_a and i < len(a.genes):
                    genes.append(a.genes[i])
                elif not pick_a and i < len(b.genes):
                    genes.append(b.genes[i])
            if not genes:
                genes = list(a.genes) or list(b.genes)
        return DNA(genes=list(genes))

    # ---- Mutation ----
    def _mutate_gene_param(self, g: Gene) -> Gene:
        def clip01(x):
            return max(0.0, min(1.0, x))

        def clip255(x):
            return int(max(0, min(255, x)))

        # Gaussian-like small jitters
        if self.rng.random() < self.cfg.mutation_rate:
            g.x = clip01(g.x + self.rng.uniform(-0.07, 0.07))
        if self.rng.random() < self.cfg.mutation_rate:
            g.y = clip01(g.y + self.rng.uniform(-0.07, 0.07))
        if self.rng.random() < self.cfg.mutation_rate:
            g.w = clip01(g.w * (1.0 + self.rng.uniform(-0.25, 0.25)))
        if self.rng.random() < self.cfg.mutation_rate:
            g.h = clip01(g.h * (1.0 + self.rng.uniform(-0.25, 0.25)))
        if self.rng.random() < self.cfg.mutation_rate:
            g.z = clip01(g.z + self.rng.uniform(-0.2, 0.2))
        if self.rng.random() < self.cfg.mutation_rate:
            g.r = clip255(g.r + int(self.rng.uniform(-32, 32)))
        if self.rng.random() < self.cfg.mutation_rate:
            g.g = clip255(g.g + int(self.rng.uniform(-32, 32)))
        if self.rng.random() < self.cfg.mutation_rate:
            g.b = clip255(g.b + int(self.rng.uniform(-32, 32)))
        if self.rng.random() < self.cfg.mutation_rate:
            g.a = clip255(g.a + int(self.rng.uniform(-40, 40)))
        if self.rng.random() < self.cfg.mutation_rate / 4.0:
            g.shape_type = "RECT" if g.shape_type == "ELLIPSE" else "ELLIPSE"
        return g

    def _mutate(self, dna: DNA) -> DNA:
        genes = [self._mutate_gene_param(Gene(**asdict(g))) for g in dna.genes]
        # structural mutations
        if (
            self.rng.random() < self.cfg.add_gene_prob
            and len(genes) < self.cfg.genes_max
        ):
            genes.append(Gene.random(self.rng))
        if (
            self.rng.random() < self.cfg.remove_gene_prob
            and len(genes) > self.cfg.genes_min
        ):
            idx = self.rng.randrange(len(genes))
            del genes[idx]
        return DNA(genes=genes)

    # ---- Next generation ----
    def next_generation(self, population: List[Tuple[DNA, float]]) -> List[DNA]:
        population.sort(key=lambda t: t[1], reverse=True)

        if self.cfg.crossover == "elite_strategy":
            return self._elite_strategy_generation(population)
        else:
            return self._standard_generation(population)

    def _elite_strategy_generation(
        self, population: List[Tuple[DNA, float]]
    ) -> List[DNA]:
        """エリート優遇戦略: 最上位個体が他のエリートと優先的に交配"""
        # 上位5個体をエリートとして保存
        elites = [
            population[i][0] for i in range(min(self.cfg.elitism, len(population)))
        ]
        new_pop: List[DNA] = [
            DNA(genes=[Gene(**asdict(g)) for g in e.genes]) for e in elites
        ]

        best_individual = elites[0] if elites else None

        # 最上位個体が他の4個体と5回交配（合計20個体）
        if best_individual and len(elites) > 1:
            for other_elite in elites[1:]:  # 他の4個体
                for _ in range(5):  # と5回交配
                    if len(new_pop) >= self.cfg.pop_size:
                        break
                    child = self._elite_crossover(best_individual, other_elite)
                    child = self._mutate(child)
                    child = self._clamp_gene_count(child)
                    new_pop.append(child)

        # 残りはエリートからランダム選択で交配
        elite_population = [
            (dna, score) for dna, score in population[: self.cfg.elitism]
        ]
        while len(new_pop) < self.cfg.pop_size:
            p1 = self._select_random_elite(elite_population)
            p2 = self._select_random_elite(elite_population)
            child = self._elite_crossover(p1, p2)
            child = self._mutate(child)
            child = self._clamp_gene_count(child)
            new_pop.append(child)

        return new_pop[: self.cfg.pop_size]

    def _standard_generation(self, population: List[Tuple[DNA, float]]) -> List[DNA]:
        """従来のトーナメント選択戦略"""
        elites = [
            population[i][0] for i in range(min(self.cfg.elitism, len(population)))
        ]
        new_pop: List[DNA] = [
            DNA(genes=[Gene(**asdict(g)) for g in e.genes]) for e in elites
        ]
        while len(new_pop) < self.cfg.pop_size:
            p1 = self._tournament(population)
            p2 = self._tournament(population)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            child = self._clamp_gene_count(child)
            new_pop.append(child)
        return new_pop[: self.cfg.pop_size]

    def _select_random_elite(self, elite_population: List[Tuple[DNA, float]]) -> DNA:
        """エリートからランダム選択"""
        return self.rng.choice(elite_population)[0]

    def _elite_crossover(self, parent1: DNA, parent2: DNA) -> DNA:
        """エリート用の交配: 3000個の遺伝子を被りなく選択"""
        target_gene_count = self.cfg.genes_min  # 3000
        all_genes = parent1.genes + parent2.genes

        if len(all_genes) >= target_gene_count:
            selected_genes = self.rng.sample(all_genes, target_gene_count)
        else:
            # 不足分はランダム生成
            selected_genes = all_genes[:]
            while len(selected_genes) < target_gene_count:
                selected_genes.append(Gene.random(self.rng))

        return DNA(genes=selected_genes)

    def _clamp_gene_count(self, dna: DNA) -> DNA:
        """遺伝子数を[min,max]に制限"""
        if len(dna.genes) < self.cfg.genes_min:
            dna.genes.extend(
                [
                    Gene.random(self.rng)
                    for _ in range(self.cfg.genes_min - len(dna.genes))
                ]
            )
        if len(dna.genes) > self.cfg.genes_max:
            dna.genes = dna.genes[: self.cfg.genes_max]
        return dna


# =====================
# Engine (FastAPI-friendly)
# =====================


class Engine:
    def __init__(
        self,
        ga_cfg: GAConfig,
        render_cfg: RenderConfig,
        prompts: List[str],
        seed: int = 42,
        scorer_model: str = "openai/clip-vit-base-patch32",
        dummy: bool = False,
        verbose: bool = False,
    ):
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.renderer = Renderer(render_cfg.width, render_cfg.height, verbose=verbose)
        self.ga = GeneticAlgorithm(ga_cfg, self.rng)
        self.timing_stats = {
            "evaluate_total": 0.0,
            "evaluate_count": 0,
            "step_total": 0.0,
            "step_count": 0,
        }
        self.population: List[DNA] = [
            DNA.random(
                self.ga.cfg.genes_min
                + self.rng.randint(
                    0, max(0, self.ga.cfg.genes_max - self.ga.cfg.genes_min)
                ),
                self.rng,
            )
            for _ in range(self.ga.cfg.pop_size)
        ]
        if dummy or not _HAS_CLIP:
            self.scorer = DummyScorer(seed=seed)
        else:
            self.scorer = CLIPScorer(model_name=scorer_model, verbose=verbose)
        self.set_prompts(prompts)
        self.generation = 0
        self.best: Optional[Tuple[DNA, float]] = None

    def set_prompts(self, prompts: List[str]):
        self.prompts = list(prompts)
        self.scorer.set_prompts(self.prompts)

    def evaluate(
        self, batch_size: int = 16, render_workers: int = 4
    ) -> List[Tuple[DNA, float]]:
        start_time = time.time()

        # Parallel rendering
        render_start = time.time()
        images = self.renderer.render_batch(self.population, n_workers=render_workers)
        render_time = time.time() - render_start

        # CLIP scoring
        score_start = time.time()
        scores = self.scorer.score(images, batch_size=batch_size)
        score_time = time.time() - score_start

        pop_scored = list(zip(self.population, scores.tolist()))
        # track best
        pop_scored.sort(key=lambda t: t[1], reverse=True)
        if self.best is None or pop_scored[0][1] > self.best[1]:
            self.best = (pop_scored[0][0], pop_scored[0][1])

        end_time = time.time()
        total_time = end_time - start_time
        self.timing_stats["evaluate_total"] += total_time
        self.timing_stats["evaluate_count"] += 1

        if self.verbose:
            print(f"Evaluate gen {self.generation}:")
            print(f"  Render time: {render_time:.4f}s")
            print(f"  CLIP time: {score_time:.4f}s")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Best score: {pop_scored[0][1]:.4f}")

        return pop_scored

    def step(self, batch_size: int = 16, render_workers: int = 4):
        start_time = time.time()

        scored = self.evaluate(batch_size=batch_size, render_workers=render_workers)

        ga_start = time.time()
        self.population = self.ga.next_generation(scored)
        ga_time = time.time() - ga_start

        self.generation += 1

        end_time = time.time()
        step_time = end_time - start_time
        self.timing_stats["step_total"] += step_time
        self.timing_stats["step_count"] += 1

        if self.verbose:
            print(f"  GA time: {ga_time:.4f}s")
            print(f"  Step total: {step_time:.4f}s")

        return scored

    def get_timing_stats(self):
        """Get comprehensive timing statistics"""
        stats = {
            "engine": self.timing_stats.copy(),
            "renderer": self.renderer.get_timing_stats(),
            "scorer": (
                self.scorer.get_timing_stats()
                if hasattr(self.scorer, "get_timing_stats")
                else {}
            ),
        }

        # Calculate averages for engine stats
        if stats["engine"]["evaluate_count"] > 0:
            stats["engine"]["avg_evaluate_time"] = (
                stats["engine"]["evaluate_total"] / stats["engine"]["evaluate_count"]
            )
        if stats["engine"]["step_count"] > 0:
            stats["engine"]["avg_step_time"] = (
                stats["engine"]["step_total"] / stats["engine"]["step_count"]
            )

        return stats

    # --- Saving helpers ---
    def save_generation(
        self, scored: List[Tuple[DNA, float]], out_dir: str, top_k: int = 6
    ):
        gen_dir = os.path.join(out_dir, f"gen_{self.generation:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        # Save top-k images + genomes
        for i, (dna, score) in enumerate(scored[:top_k]):
            img = self.renderer.render(dna)
            img.save(os.path.join(gen_dir, f"rank_{i+1:02d}_score_{score:.4f}.png"))
            with open(
                os.path.join(gen_dir, f"rank_{i+1:02d}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {"score": float(score), "genes": [asdict(g) for g in dna.genes]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        # Save brief metrics
        metrics = {
            "generation": self.generation,
            "prompts": self.prompts,
            "scores_top": [float(s) for _, s in scored[:top_k]],
            "scores_mean": float(np.mean([s for _, s in scored])),
            "scores_max": float(scored[0][1]),
            "population_size": len(scored),
        }
        with open(os.path.join(gen_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_best(self, out_dir: str):
        if not self.best:
            return
        best_dir = os.path.join(out_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        dna, score = self.best
        img = self.renderer.render(dna)
        img.save(os.path.join(best_dir, f"best_score_{score:.4f}.png"))
        with open(os.path.join(best_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"score": float(score), "genes": [asdict(g) for g in dna.genes]},
                f,
                ensure_ascii=False,
                indent=2,
            )


# =====================
# Display Helper
# =====================


class EvolutionDisplay:
    def __init__(self, engine: Engine, show_top_k: int = 6):
        if not _HAS_DISPLAY:
            raise RuntimeError(
                "matplotlib not available. Install matplotlib to use display."
            )
        self.engine = engine
        self.show_top_k = show_top_k
        self.fig = None
        self.axes = None
        self.images = []

    def setup(self):
        plt.ion()  # Interactive mode
        cols = min(3, self.show_top_k)
        rows = (self.show_top_k + cols - 1) // cols
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.array([self.axes])
        elif cols == 1:
            self.axes = self.axes[:, np.newaxis]
        self.fig.suptitle(f"Generation: 0 | Prompts: {', '.join(self.engine.prompts)}")

        # Initialize empty plots
        for idx in range(rows * cols):
            ax = self.axes[idx // cols, idx % cols]
            ax.axis("off")
            if idx < self.show_top_k:
                ax.set_title(f"Rank {idx+1}", fontsize=10)

        plt.tight_layout()

    def update(self, scored: List[Tuple[DNA, float]]):
        if self.fig is None:
            self.setup()

        # Update title
        self.fig.suptitle(
            f"Generation: {self.engine.generation} | Best Score: {scored[0][1]:.4f} | Prompts: {', '.join(self.engine.prompts)}"
        )

        # Update images
        for idx in range(self.show_top_k):
            ax = self.axes[idx // 3, idx % 3]
            ax.clear()
            ax.axis("off")

            if idx < len(scored):
                dna, score = scored[idx]
                img = self.engine.renderer.render(dna)
                ax.imshow(img)
                ax.set_title(f"Rank {idx+1} | Score: {score:.4f}", fontsize=10)

        plt.draw()
        plt.pause(0.001)  # Small pause to update display

    def close(self):
        if self.fig:
            plt.close(self.fig)


# =====================
# CLI
# =====================


def main():
    p = argparse.ArgumentParser(description="Evolve abstract images scored by CLIP.")
    p.add_argument("--config", type=str, help="path to YAML config file")
    p.add_argument(
        "--create-config",
        type=str,
        help="create a default config file at the specified path",
    )
    p.add_argument("--out", type=str, help="output directory (overrides config)")
    p.add_argument(
        "--prompts",
        nargs="+",
        help="one or more prompts (overrides config)",
    )
    p.add_argument(
        "--generations", type=int, help="number of generations (overrides config)"
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="show evolution in real-time (overrides config)",
    )
    p.add_argument(
        "--dummy", action="store_true", help="use random scorer (overrides config)"
    )
    args = p.parse_args()

    # Create default config if requested
    if args.create_config:
        config = Config()  # Default config
        config.to_file(args.create_config)
        print(f"Default config created at: {args.create_config}")
        return

    # Load config from file or use defaults
    if args.config:
        config = Config.from_file(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = Config()  # Default config
        print(
            "Using default configuration. Use --create-config to save defaults to file."
        )

    # Override config with command line arguments
    if args.out:
        config.run.out_dir = args.out
    if args.prompts:
        config.model.prompts = args.prompts
    if args.generations:
        config.run.generations = args.generations
    if args.show:
        config.run.show = True
    if args.dummy:
        config.model.dummy = True

    # Use configs from loaded/default config
    ga_cfg = config.ga
    render_cfg = config.render

    # Init engine
    engine = Engine(
        ga_cfg=ga_cfg,
        render_cfg=render_cfg,
        prompts=config.model.prompts,
        seed=ga_cfg.seed,
        scorer_model=config.model.model,
        dummy=config.model.dummy,
    )

    out = config.run.out_dir
    os.makedirs(out, exist_ok=True)

    # Setup display if requested
    display = None
    if config.run.show:
        if not _HAS_DISPLAY:
            print(
                "Warning: matplotlib not available. Install with: pip install matplotlib"
            )
            print("Continuing without display...")
        else:
            display = EvolutionDisplay(engine, show_top_k=config.run.show_top_k)

    # Save initial random population top-k (generation 0)
    scored = engine.evaluate(
        batch_size=config.run.batch_size, render_workers=config.run.render_workers
    )
    # Save generation 0 (always save initial)
    engine.save_generation(scored, out_dir=out, top_k=config.run.save_top_k)
    engine.save_best(out)

    if display:
        display.update(scored)

    # Evolve
    for gen in tqdm(range(config.run.generations), desc="Evolving"):
        scored = engine.step(
            batch_size=config.run.batch_size, render_workers=config.run.render_workers
        )

        # Save every save_freq generations, or if it's the last generation
        if (engine.generation % config.run.save_freq == 0) or (
            gen == config.run.generations - 1
        ):
            engine.save_generation(scored, out_dir=out, top_k=config.run.save_top_k)

        # Always update best (lightweight operation)
        engine.save_best(out)

        if display:
            display.update(scored)

    print("Done. Results in:", out)

    if display:
        display.close()
        # Keep window open for a moment
        if _HAS_DISPLAY:
            plt.show(block=True)


if __name__ == "__main__":
    main()
