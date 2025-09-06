# GA-CLIP Engine (遺伝アルゴリズム × CLIP評価)

talklog.txtの会話をベースに作成した、遺伝アルゴリズムで抽象画像を進化させ、CLIPで自動評価するシステムです。
FastAPIへの拡張を想定したモジュール構成になっています。

## 特徴

- **遺伝子表現**: 半透明の図形（長方形/楕円）のリストとして画像を表現
- **YAML設定**: 設定ファイルによる簡単なパラメータ管理
- **エリート戦略**: 参考文献ベースの最適化された交配システム
- **レンダリング**: Pillowでアルファ合成により画像生成
- **CLIP評価**: transformersの`openai/clip-vit-base-patch32`でテキストプロンプトとの類似度を計算
- **遺伝アルゴリズム**: エリート優遇戦略、パラメータ・構造突然変異
- **拡張性**: Engineクラスで統合、FastAPI化が容易

## 遺伝子構造

### Gene（遺伝子）の表現

各個体は複数の「Gene」で構成され、1つのGeneが1つの図形を表現します：

```python
@dataclass
class Gene:
    shape_type: str    # 図形タイプ："RECT"（長方形）または "ELLIPSE"（楕円）
    x: float          # X座標 (0.0-1.0、画像幅に対する相対位置)
    y: float          # Y座標 (0.0-1.0、画像高さに対する相対位置)
    w: float          # 幅 (0.0-1.0、画像幅に対する相対サイズ)
    h: float          # 高さ (0.0-1.0、画像高さに対する相対サイズ)
    z: float          # 描画順序 (0.0-1.0、小さいほど背景側)
    r: int            # 赤色成分 (0-255)
    g: int            # 緑色成分 (0-255)  
    b: int            # 青色成分 (0-255)
    a: int            # 透明度 (0-255、0=完全透明、255=不透明)
```

### DNA（個体）の構成

```python
@dataclass
class DNA:
    genes: List[Gene]  # 複数の図形（遺伝子）のリスト
```

### 遺伝子数の制御

- **genes_min/max**: 個体が持つ遺伝子数の範囲（図形数）
- **add_gene_prob**: 突然変異時に遺伝子を追加する確率（1%推奨）
- **remove_gene_prob**: 突然変異時に遺伝子を削除する確率（1%推奨）

```yaml
# 推奨設定：3000個固定（参考文献ベース）
ga:
  genes_min: 3000
  genes_max: 3000
  add_gene_prob: 0.01    # 低確率で構造安定性を確保
  remove_gene_prob: 0.01 # パラメータ最適化に集中

# 可変設定例（探索重視）
ga:
  genes_min: 500
  genes_max: 4000
  add_gene_prob: 0.03
  remove_gene_prob: 0.02
```

**低確率設定の意図**：
- 構造変化による性能ブレを防止
- パラメータ調整に集中して品質向上
- 一定遺伝子数で処理時間を予測可能に

### 画像生成プロセス

1. **ソート**: 全遺伝子をz値（描画順序）でソート
2. **描画**: 背景から順番に各図形を半透明で重ね描き
3. **合成**: アルファブレンディングで最終画像を生成

```python
# 描画プロセスの例
for gene in sorted(dna.genes, key=lambda x: x.z):
    # 図形の実際の座標・サイズを計算
    cx = int(gene.x * 画像幅)
    cy = int(gene.y * 画像高さ)
    w = int(gene.w * 画像幅)
    h = int(gene.h * 画像高さ)
    
    # 色と透明度
    color = (gene.r, gene.g, gene.b, gene.a)
    
    # 図形を描画して合成
    if gene.shape_type == "RECT":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:  # ELLIPSE
        draw.ellipse([x0, y0, x1, y1], fill=color)
```

## アルゴリズム概要

### 進化戦略の選択

本システムでは2つの進化戦略から選択できます：

#### エリート優遇戦略（推奨）

高速収束を重視した戦略：

1. **エリート保存**: 上位5個体を無条件で次世代に保存
2. **優遇交配**: 最上位個体が他の4個体と各5回交配（20個体生成）
3. **エリート交配**: 残りはエリートからランダム選択で交配

```python
# エリート交配の核心アルゴリズム
def elite_crossover(parent1, parent2):
    all_genes = parent1.genes + parent2.genes
    selected_genes = random.sample(all_genes, 3000)  # 被りなく3000個選択
    return DNA(genes=selected_genes)
```

#### スタンダード戦略（トーナメント選択）

探索重視のバランス型戦略：

1. **エリート保存**: 上位5個体を保存
2. **トーナメント選択**: 4個体から最良を選択（平等な機会）
3. **一様交配**: 各遺伝子を50%の確率で選択

```python
# スタンダード戦略の流れ
def standard_generation(population):
    elites = population[:5]  # エリート保存
    new_pop = elites[:]
    
    while len(new_pop) < pop_size:
        p1 = tournament_selection(population, k=4)
        p2 = tournament_selection(population, k=4)
        child = uniform_crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)
```

#### 戦略比較

| 項目         | **エリート優遇**              | **スタンダード**       |
| ------------ | ----------------------------- | ---------------------- |
| **収束速度** | 高速                          | 穏やか                 |
| **多様性**   | 低め（最上位重視）            | 高め（平等選択）       |
| **探索能力** | 局所最適化                    | 広域探索               |
| **適用場面** | 短時間で高品質                | じっくり探索           |
| **設定**     | `crossover: "elite_strategy"` | `crossover: "uniform"` |

### 突然変異戦略

各遺伝子に対して12%の確率で以下の変異を適用：

| 対象        | 変化範囲    | 効果               |
| ----------- | ----------- | ------------------ |
| 位置(x,y)   | ±7%         | 図形の微調整移動   |
| サイズ(w,h) | ±25%        | 図形の拡大縮小     |
| 色(RGB)     | ±32/255     | 色の微調整         |
| 透明度(α)   | ±40/255     | 透明度の調整       |
| 形状        | 長方形⇔楕円 | 3%の確率で変更     |
| 構造        | ±1遺伝子    | 1%の確率で個数変化 |

### CLIP評価

```python
# 適応度計算プロセス
image = render(individual)                    # 個体を画像化
image_features = clip_model.encode_image(image)
text_features = clip_model.encode_text(prompts)
fitness = cosine_similarity(image_features, text_features)
```

## インストール

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windowsは venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

## 使い方

### 1. 設定ファイルを使用した実行（推奨）

```bash
# デフォルト設定ファイルを作成
python ga_clip_engine.py --create-config my_config.yaml

# 設定ファイルで実行
python ga_clip_engine.py --config my_config.yaml

# 設定の一部をコマンドラインで上書き
python ga_clip_engine.py --config my_config.yaml --generations 100 --prompts "beautiful dog"

# 既存出力ディレクトリの処理オプション
python ga_clip_engine.py --config my_config.yaml --overwrite     # 自動削除
python ga_clip_engine.py --config my_config.yaml --auto-suffix   # 自動suffix追加
```

### 2. 直接実行（設定ファイルなし）

```bash
# CLIPなしで動作確認（ダミースコア）
python ga_clip_engine.py --dummy --generations 5

# CLIPで評価して進化
python ga_clip_engine.py --prompts "aesthetic curves" --generations 50
```

### 設定ファイル例

#### エリート優遇戦略（高速収束）
```yaml
# config_elite.yaml
ga:
  pop_size: 40
  genes_min: 3000
  genes_max: 3000
  elitism: 5
  crossover: "elite_strategy"     # エリート優遇戦略

run:
  generations: 500                # 短期間で収束
  out_dir: "runs/elite_experiment"

model:
  prompts: ["beautiful landscape"]
  dummy: false
```

#### スタンダード戦略（探索重視）
```yaml
# config_standard.yaml
ga:
  pop_size: 40
  genes_min: 3000
  genes_max: 3000
  elitism: 5
  crossover: "uniform"            # スタンダード戦略

run:
  generations: 2000               # 長期間でじっくり探索
  out_dir: "runs/standard_experiment"

model:
  prompts: ["beautiful landscape"]
  dummy: false
```

### パラメータ説明

**遺伝的アルゴリズム（ga）**:
- `pop_size`: 個体数（40推奨）
- `genes_min/max`: 図形数の最小/最大（3000固定推奨）
- `elitism`: エリート保存数（5推奨）
- `crossover`: 交配戦略（"elite_strategy"推奨）
- `mutation_rate`: 突然変異率（0.12推奨）
- `seed`: 乱数シード（nullの場合はunix timeから自動生成）

**実行設定（run）**:
- `generations`: 世代数（2000推奨、テストは10-100）
- `out_dir`: 出力ディレクトリ
- `save_freq`: 保存頻度（N世代毎）
- `show`: リアルタイム表示（true/false）

**モデル設定（model）**:
- `prompts`: 評価用テキストプロンプト（リスト）
- `model`: CLIPモデル名
- `dummy`: ダミースコア使用（テスト用）

## 出力

実行すると指定ディレクトリに以下が保存されます：

```
runs/demo/
├── config.yaml                    # 実行時の設定ファイル
├── config_with_actual_seed.yaml   # 実際に使用されたseed付き設定（再現性確保）
├── gen_000/                        # 各世代のトップ画像とメトリクス
│   ├── rank_01_score_0.xxxx.png
│   ├── rank_01.json                # 遺伝子情報
│   └── metrics.json                # 世代統計
├── gen_001/
├── ...
└── best/                          # 最高スコアの個体
    ├── best_score_0.xxxx_15.png   # スコア_世代数
    └── best.json
```

## FastAPI拡張例

```python
from fastapi import FastAPI
from ga_clip_engine import Engine, GAConfig, RenderConfig

app = FastAPI()
engine = Engine(GAConfig(), RenderConfig(), prompts=["aesthetic art"])

@app.post("/prompts")
def set_prompts(prompts: list[str]):
    engine.set_prompts(prompts)
    return {"status": "ok", "prompts": prompts}

@app.post("/step")
def evolve_one_generation():
    scored = engine.step()
    return {
        "generation": engine.generation,
        "best_score": float(max(s for _, s in scored)),
        "mean_score": float(sum(s for _, s in scored) / len(scored))
    }

@app.get("/population/{index}")
def get_individual_image(index: int):
    if 0 <= index < len(engine.population):
        img = engine.renderer.render(engine.population[index])
        # Return as base64 or save to temp file
        return {"index": index, "genes": len(engine.population[index].genes)}
    return {"error": "Index out of range"}
```

## プロンプト例

- `"aesthetic curves"` - 美的な曲線
- `"vibrant colors"` - 鮮やかな色彩
- `"minimalist composition"` - ミニマリストな構成
- `"abstract art"` - 抽象芸術
- `"geometric patterns"` - 幾何学パターン
- `"soft gradients"` - 柔らかなグラデーション
- `"1dog"` - 犬の画像
- `"beautiful landscape"` - 美しい風景
- `"modern art style"` - モダンアートスタイル


```bash
# 高速テスト
python ga_clip_engine.py --config config_fast.yaml

# バランス実験
python ga_clip_engine.py --config config_balanced.yaml

# 高品質実験
python ga_clip_engine.py --config config.yaml
```

## 注意事項

- GPUがあれば`torch`がCUDA版になっているか確認してください
- 初回実行時はCLIPモデルのダウンロードに時間がかかります