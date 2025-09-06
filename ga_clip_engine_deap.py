#!/usr/bin/env python3
"""
DEAP (Distributed Evolutionary Algorithms in Python)を使用したGA-CLIP絵画進化システム

オリジナルのga_clip_engine.pyと同じ機能を持ちながら、
DEAPライブラリの豊富な進化戦略とオペレータを活用
"""
from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
import shutil
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Iterable, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# DEAP imports
try:
    from deap import base, creator, tools, algorithms
    import deap

    _HAS_DEAP = True
except ImportError:
    _HAS_DEAP = False

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
# Genome & Rendering (既存のコードを再利用)
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

    def to_list(self) -> List[float]:
        """DEAP用にパラメータをリスト形式で返す"""
        return [
            float(SHAPE_TYPES.index(self.shape_type)),
            self.x,
            self.y,
            self.w,
            self.h,
            self.z,
            float(self.r),
            float(self.g),
            float(self.b),
            float(self.a),
        ]

    @classmethod
    def from_list(cls, values: List[float]) -> "Gene":
        """リスト形式からGeneオブジェクトを作成"""
        return cls(
            shape_type=SHAPE_TYPES[int(values[0]) % len(SHAPE_TYPES)],
            x=values[1],
            y=values[2],
            w=values[3],
            h=values[4],
            z=values[5],
            r=int(values[6]),
            g=int(values[7]),
            b=int(values[8]),
            a=int(values[9]),
        )


@dataclass
class DNA:
    genes: List[Gene]

    @staticmethod
    def random(gene_count: int, rng: random.Random) -> "DNA":
        return DNA(genes=[Gene.random(rng) for _ in range(gene_count)])

    def to_individual(self) -> List[float]:
        """DEAP個体（フラットなリスト）に変換"""
        individual = []
        for gene in self.genes:
            individual.extend(gene.to_list())
        return individual

    @classmethod
    def from_individual(
        cls, individual: List[float], genes_per_individual: int
    ) -> "DNA":
        """DEAP個体からDNAオブジェクトを作成"""
        genes = []
        gene_size = 10  # Geneのパラメータ数
        for i in range(0, len(individual), gene_size):
            if len(genes) >= genes_per_individual:
                break
            gene_params = individual[i : i + gene_size]
            if len(gene_params) == gene_size:
                genes.append(Gene.from_list(gene_params))
        return cls(genes=genes)


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
# CLIP Scorer (既存のコードを再利用)
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
            try:
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
# Configuration (既存のコードを再利用)
# =====================


@dataclass
class GAConfig:
    pop_size: int = 40
    genes_min: int = 3000
    genes_max: int = 3000
    elitism: int = 5
    tournament_k: int = 4
    mutation_rate: float = 0.12
    add_gene_prob: float = 0.01
    remove_gene_prob: float = 0.01
    crossover: str = "uniform"  # DEAP用に調整
    mutation_strength: float = 0.1  # DEAP用パラメータ
    crossover_prob: float = 0.8  # DEAP用パラメータ
    mutation_prob: float = 0.2  # DEAP用パラメータ
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "GAConfig":
        # dataclassのフィールド名のリストを取得
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


@dataclass
class RenderConfig:
    width: int = 256
    height: int = 256

    @classmethod
    def from_dict(cls, data: dict) -> "RenderConfig":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


@dataclass
class RunConfig:
    generations: int = 30
    save_top_k: int = 6
    out_dir: str = "runs/demo_deap"
    batch_size: int = 16
    render_workers: int = 8
    save_freq: int = 100
    show: bool = False
    show_top_k: int = 6

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


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
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


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


# =====================
# DEAP Setup and Operators
# =====================


def setup_deap(ga_config: GAConfig, gene_count: int):
    """DEAP環境の初期化"""
    if not _HAS_DEAP:
        raise RuntimeError("DEAP not available. Install with: pip install deap")

    # クリーンアップ（既存の定義を削除）
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    # フィットネス関数の定義（最大化問題）
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # 個体の定義（フラットなリスト）
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Toolboxの設定
    toolbox = base.Toolbox()

    # 遺伝子のパラメータ数
    gene_param_count = 10  # Gene のパラメータ数
    individual_size = gene_count * gene_param_count

    # 個体の初期化
    def create_individual():
        # ランダムなDNAを作成し、個体に変換
        dna = DNA.random(gene_count, random.Random())
        return creator.Individual(dna.to_individual())

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 変異オペレータ
    def mutate_individual(individual, mutation_rate=0.1):
        """個体の変異"""
        gene_param_count = 10
        gene_count = len(individual) // gene_param_count

        for i in range(0, len(individual), gene_param_count):
            if i + gene_param_count > len(individual):
                break

            # 各遺伝子パラメータに対して変異を適用
            gene_params = individual[i : i + gene_param_count]

            # shape_type (0)
            if random.random() < mutation_rate / 4.0:
                gene_params[0] = float(random.randint(0, len(SHAPE_TYPES) - 1))

            # x, y (1, 2) - 位置
            if random.random() < mutation_rate:
                gene_params[1] = max(
                    0.0, min(1.0, gene_params[1] + random.uniform(-0.07, 0.07))
                )
            if random.random() < mutation_rate:
                gene_params[2] = max(
                    0.0, min(1.0, gene_params[2] + random.uniform(-0.07, 0.07))
                )

            # w, h (3, 4) - サイズ
            if random.random() < mutation_rate:
                gene_params[3] = max(
                    0.01, min(1.0, gene_params[3] * (1.0 + random.uniform(-0.25, 0.25)))
                )
            if random.random() < mutation_rate:
                gene_params[4] = max(
                    0.01, min(1.0, gene_params[4] * (1.0 + random.uniform(-0.25, 0.25)))
                )

            # z (5) - 深度
            if random.random() < mutation_rate:
                gene_params[5] = max(
                    0.0, min(1.0, gene_params[5] + random.uniform(-0.2, 0.2))
                )

            # r, g, b, a (6, 7, 8, 9) - 色
            if random.random() < mutation_rate:
                gene_params[6] = max(
                    0.0, min(255.0, gene_params[6] + random.uniform(-32, 32))
                )
            if random.random() < mutation_rate:
                gene_params[7] = max(
                    0.0, min(255.0, gene_params[7] + random.uniform(-32, 32))
                )
            if random.random() < mutation_rate:
                gene_params[8] = max(
                    0.0, min(255.0, gene_params[8] + random.uniform(-32, 32))
                )
            if random.random() < mutation_rate:
                gene_params[9] = max(
                    30.0, min(200.0, gene_params[9] + random.uniform(-40, 40))
                )

            # 更新
            individual[i : i + gene_param_count] = gene_params

        return (individual,)

    # 交叉オペレータ
    def crossover_two_point(ind1, ind2):
        """2点交叉"""
        size = min(len(ind1), len(ind2))
        if size < 2:
            return ind1, ind2

        pt1 = random.randint(1, size - 1)
        pt2 = random.randint(1, size - 1)
        if pt1 > pt2:
            pt1, pt2 = pt2, pt1

        ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
        return ind1, ind2

    # オペレータの登録
    toolbox.register("mate", crossover_two_point)
    toolbox.register("mutate", mutate_individual, mutation_rate=ga_config.mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=ga_config.tournament_k)

    return toolbox


# =====================
# DEAP Engine
# =====================


class DEAPEngine:
    """DEAP を使用した進化エンジン"""

    def __init__(
        self,
        ga_cfg: GAConfig,
        render_cfg: RenderConfig,
        prompts: List[str],
        seed: Optional[int] = None,
        scorer_model: str = "openai/clip-vit-base-patch32",
        dummy: bool = False,
        verbose: bool = False,
    ):
        # seedが指定されていない場合はunix timeから生成
        if seed is None:
            seed = int(time.time())
            print(f"Auto-generated seed from unix time: {seed}")
        self.actual_seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.ga_cfg = ga_cfg
        self.verbose = verbose
        self.renderer = Renderer(render_cfg.width, render_cfg.height, verbose=verbose)

        # DEAP setup
        self.gene_count = ga_cfg.genes_min
        self.toolbox = setup_deap(ga_cfg, self.gene_count)

        # 統計情報
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # Hall of Fame（最優秀個体の保存）
        self.hof = tools.HallOfFame(1)

        # 集団の初期化
        self.population = self.toolbox.population(n=ga_cfg.pop_size)

        # CLIP Scorer
        if dummy or not _HAS_CLIP:
            self.scorer = DummyScorer(seed=seed)
        else:
            self.scorer = CLIPScorer(model_name=scorer_model, verbose=verbose)
        self.set_prompts(prompts)

        self.generation = 0
        self.timing_stats = {
            "evaluate_total": 0.0,
            "evaluate_count": 0,
            "step_total": 0.0,
            "step_count": 0,
        }

        # 最初の評価
        self.evaluate_population()

    def set_prompts(self, prompts: List[str]):
        self.prompts = list(prompts)
        self.scorer.set_prompts(self.prompts)

    def individual_to_dna(self, individual) -> DNA:
        """DEAP個体をDNAオブジェクトに変換"""
        return DNA.from_individual(individual, self.gene_count)

    def evaluate_population(self, batch_size: int = 16, render_workers: int = 4):
        """集団全体の評価"""
        start_time = time.time()

        # 未評価の個体のみを評価
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]

        if invalid_ind:
            # DNAに変換
            dnas = [self.individual_to_dna(ind) for ind in invalid_ind]

            # レンダリング
            render_start = time.time()
            images = self.renderer.render_batch(dnas, n_workers=render_workers)
            render_time = time.time() - render_start

            # CLIP評価
            score_start = time.time()
            scores = self.scorer.score(images, batch_size=batch_size)
            score_time = time.time() - score_start

            # フィットネス値を設定
            for ind, score in zip(invalid_ind, scores):
                ind.fitness.values = (float(score),)

        # 統計更新
        record = self.stats.compile(self.population)
        self.hof.update(self.population)

        end_time = time.time()
        total_time = end_time - start_time
        self.timing_stats["evaluate_total"] += total_time
        self.timing_stats["evaluate_count"] += 1

        if self.verbose and invalid_ind:
            print(f"Evaluate gen {self.generation}:")
            print(f"  Evaluated {len(invalid_ind)} individuals")
            print(f"  Render time: {render_time:.4f}s")
            print(f"  CLIP time: {score_time:.4f}s")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Stats: {record}")

        return record

    def step(self, batch_size: int = 16, render_workers: int = 4):
        """1世代の進化ステップ"""
        start_time = time.time()

        # 選択
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = [self.toolbox.clone(ind) for ind in offspring]

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.ga_cfg.crossover_prob:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 変異
        for mutant in offspring:
            if random.random() < self.ga_cfg.mutation_prob:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # エリート保存
        if self.ga_cfg.elitism > 0:
            # 現在の集団をスコアでソート
            sorted_pop = sorted(
                self.population, key=lambda x: x.fitness.values[0], reverse=True
            )
            elites = sorted_pop[: self.ga_cfg.elitism]

            # エリートを次世代に含める
            offspring[-self.ga_cfg.elitism :] = [
                self.toolbox.clone(elite) for elite in elites
            ]

        self.population[:] = offspring
        self.generation += 1

        # 評価
        record = self.evaluate_population(
            batch_size=batch_size, render_workers=render_workers
        )

        end_time = time.time()
        step_time = end_time - start_time
        self.timing_stats["step_total"] += step_time
        self.timing_stats["step_count"] += 1

        if self.verbose:
            print(f"  Step total: {step_time:.4f}s")

        return record

    def get_scored_population(self) -> List[Tuple[DNA, float]]:
        """現在の集団をスコア付きで返す"""
        scored = []
        for ind in self.population:
            dna = self.individual_to_dna(ind)
            score = ind.fitness.values[0] if ind.fitness.valid else 0.0
            scored.append((dna, score))

        # スコアでソート
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_best(self) -> Optional[Tuple[DNA, float]]:
        """最優秀個体を返す"""
        if self.hof:
            best_ind = self.hof[0]
            best_dna = self.individual_to_dna(best_ind)
            best_score = best_ind.fitness.values[0]
            return (best_dna, best_score)
        return None

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
    def save_generation(self, out_dir: str, top_k: int = 6):
        """世代の保存"""
        scored = self.get_scored_population()
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
        """最優秀個体の保存"""
        best = self.get_best()
        if not best:
            return

        dna, score = best
        score_str = f"{score:.4f}"

        best_dir = os.path.join(out_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        img = self.renderer.render(dna)
        img.save(
            os.path.join(best_dir, f"best_score_{score_str}_{self.generation}.png")
        )
        with open(os.path.join(best_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "score": float(score),
                    "genes": [asdict(g) for g in dna.genes],
                    "generation": self.generation,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


# =====================
# Display Helper (既存のコードを再利用)
# =====================


class EvolutionDisplay:
    def __init__(self, engine: DEAPEngine, show_top_k: int = 6):
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
# Helper Functions (既存のコードを再利用)
# =====================


def handle_existing_output_dir(
    out_dir: str, overwrite: bool = False, auto_suffix: bool = False
) -> str:
    """既存の出力ディレクトリの処理"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # 非対話的オプションの処理
    if overwrite:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        print(f"既存ディレクトリを削除しました: {out_dir}")
        return out_dir

    if auto_suffix:
        base_dir = out_dir.rstrip("/")
        counter = 1
        while True:
            new_dir = f"{base_dir}_{counter}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                print(f"新しいディレクトリを作成しました: {new_dir}")
                return new_dir
            counter += 1

    # 対話的選択
    print(f"\n出力ディレクトリが既に存在します: {out_dir}")
    print("選択してください:")
    print("1. 既存ディレクトリを削除して実行")
    print("2. 新しい名前で実行 (suffix _1, _2, ...)")
    print("3. 実行を中止")

    while True:
        try:
            choice = input("選択 (1/2/3): ").strip()

            if choice == "1":
                # 既存ディレクトリを削除
                shutil.rmtree(out_dir)
                os.makedirs(out_dir)
                print(f"既存ディレクトリを削除しました: {out_dir}")
                return out_dir

            elif choice == "2":
                # 新しい名前を生成
                base_dir = out_dir.rstrip("/")
                counter = 1
                while True:
                    new_dir = f"{base_dir}_{counter}"
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                        print(f"新しいディレクトリを作成しました: {new_dir}")
                        return new_dir
                    counter += 1

            elif choice == "3":
                print("実行を中止しました。")
                exit(0)

            else:
                print("1, 2, 3のいずれかを入力してください。")

        except KeyboardInterrupt:
            print("\n実行を中止しました。")
            exit(0)


# =====================
# CLI
# =====================


def main():
    p = argparse.ArgumentParser(
        description="Evolve abstract images scored by CLIP using DEAP."
    )
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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="automatically overwrite existing output directory",
    )
    p.add_argument(
        "--auto-suffix",
        action="store_true",
        help="automatically add suffix if output directory exists",
    )
    args = p.parse_args()

    # Check DEAP availability
    if not _HAS_DEAP:
        print("Error: DEAP library is required but not installed.")
        print("Install with: pip install deap")
        return

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

    print(f"config: {config}")

    # Use configs from loaded/default config
    ga_cfg = config.ga
    render_cfg = config.render

    # Init DEAP engine
    engine = DEAPEngine(
        ga_cfg=ga_cfg,
        render_cfg=render_cfg,
        prompts=config.model.prompts,
        seed=ga_cfg.seed,  # Noneの場合はEngine内でunix timeから自動生成
        scorer_model=config.model.model,
        dummy=config.model.dummy,
        verbose=True,
    )

    out = config.run.out_dir
    out = handle_existing_output_dir(
        out, overwrite=args.overwrite, auto_suffix=args.auto_suffix
    )

    # Copy config file to output directory for reproducibility
    if args.config:
        config_dest = os.path.join(out, "config.yaml")
        shutil.copy2(args.config, config_dest)
        print(f"Config copied to: {config_dest}")
    else:
        # Save current config if no config file was used
        config_dest = os.path.join(out, "config.yaml")
        config.to_file(config_dest)
        print(f"Config saved to: {config_dest}")

    # Save actual used seed for reproducibility
    config.ga.seed = engine.actual_seed
    config_with_seed_dest = os.path.join(out, "config_with_actual_seed.yaml")
    config.to_file(config_with_seed_dest)
    print(f"Config with actual seed saved to: {config_with_seed_dest}")

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
    scored = engine.get_scored_population()
    engine.save_generation(out_dir=out, top_k=config.run.save_top_k)
    engine.save_best(out)

    if display:
        display.update(scored)

    # Evolve
    for gen in tqdm(range(config.run.generations), desc="Evolving with DEAP"):
        record = engine.step(
            batch_size=config.run.batch_size, render_workers=config.run.render_workers
        )

        scored = engine.get_scored_population()

        # Save every save_freq generations, or if it's the last generation
        if (engine.generation % config.run.save_freq == 0) or (
            gen == config.run.generations - 1
        ):
            engine.save_generation(out_dir=out, top_k=config.run.save_top_k)

        # Always update best (lightweight operation)
        engine.save_best(out)

        if display:
            display.update(scored)

    print("Done. Results in:", out)
    print(
        f"Best score achieved: {engine.get_best()[1]:.4f}"
        if engine.get_best()
        else "No best score"
    )

    # Print timing statistics
    stats = engine.get_timing_stats()
    print("\nTiming Statistics:")
    print(
        f"  Average evaluate time: {stats['engine'].get('avg_evaluate_time', 0):.4f}s"
    )
    print(f"  Average step time: {stats['engine'].get('avg_step_time', 0):.4f}s")

    if display:
        display.close()
        # Keep window open for a moment
        if _HAS_DISPLAY:
            plt.show(block=True)


if __name__ == "__main__":
    main()
