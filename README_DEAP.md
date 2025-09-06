# GA-CLIP Engine with DEAP

このドキュメントは、DEAPライブラリを使用したGA-CLIP Engine（`ga_clip_engine_deap.py`）について説明します。

## 概要

元の`ga_clip_engine.py`と同じ機能を持ちながら、[DEAP (Distributed Evolutionary Algorithms in Python)](https://github.com/DEAP/deap)ライブラリを使用して遺伝的アルゴリズムを実装したバージョンです。

## 主な特徴

### DEAPライブラリの活用
- **豊富な進化戦略**: DEAPの標準的な選択・交叉・変異オペレータを使用
- **統計管理**: 世代ごとの統計情報（平均、標準偏差、最小値、最大値）を自動収集
- **Hall of Fame**: 最優秀個体の自動管理
- **拡張性**: DEAPの豊富なオペレータライブラリにより、容易にアルゴリズムの改良が可能

### 既存実装との違い

| 項目         | オリジナル版         | DEAP版                         |
| ------------ | -------------------- | ------------------------------ |
| GA実装       | 独自実装             | DEAPライブラリ使用             |
| 個体表現     | DNA/Geneクラス       | フラットなリスト + 変換機能    |
| 統計情報     | 基本的なスコア情報   | 豊富な統計情報（平均、分散等） |
| 拡張性       | カスタムコードが必要 | DEAPの標準オペレータが利用可能 |
| エリート保存 | 独自実装             | DEAPの標準機能                 |

## インストール

DEAPライブラリが必要です：

```bash
pip install deap>=1.3.0
```

または、更新されたrequirements.txtを使用：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

```bash
# DEAPエンジンを使用して実行
python ga_clip_engine_deap.py --prompts "beautiful landscape" --generations 50
```

### 設定ファイルでの実行

```bash
# 設定ファイルを作成
python ga_clip_engine_deap.py --create-config config_deap.yaml

# 設定ファイルを使用して実行
python ga_clip_engine_deap.py --config config_deap.yaml
```

### DEAP固有の設定項目

GAConfigに以下の項目が追加されています：

```yaml
ga:
  # 既存の設定項目...
  crossover_prob: 0.8      # 交叉確率
  mutation_prob: 0.2       # 変異確率
  mutation_strength: 0.1   # 変異の強度
```

## コード構造

### 主要クラス

1. **DEAPEngine**: DEAP統合の進化エンジン
   - DEAP環境の初期化
   - 集団の管理と進化
   - 統計情報の収集

2. **Gene/DNA**: 既存のゲノム表現（オリジナルと同じ）
   - DEAP個体との相互変換機能を追加

3. **各種オペレータ**: DEAP用のカスタムオペレータ
   - `mutate_individual`: 遺伝子パラメータ特化の変異
   - `crossover_two_point`: 2点交叉

### DEAP統合のポイント

```python
# フィットネス関数の定義（最大化問題）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 個体の定義（フラットなリスト）
creator.create("Individual", list, fitness=creator.FitnessMax)

# 統計情報の収集
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
```

## パフォーマンスと最適化

### 利点
- **コード品質**: 長年テストされたDEAPライブラリの安定性
- **メンテナンス性**: 標準的な実装により保守が容易
- **拡張性**: 新しい選択・交叉・変異オペレータを簡単に試せる
- **デバッグ**: DEAPの統計機能により進化過程の詳細な分析が可能

### 注意点
- 個体の表現がフラットなリストになるため、DNA⇔Individual変換のオーバーヘッドが発生
- DEAPライブラリへの依存が追加される

## 出力と結果

出力形式はオリジナル版と同じです：
- `runs/demo_deap/` ディレクトリに結果を保存（デフォルト）
- 世代ごとのトップ画像とメタデータ
- 最優秀個体の継続的な保存
- 設定ファイルの自動保存

## トラブルシューティング

### よくある問題

1. **DEAPライブラリがインストールされていない**
   ```
   Error: DEAP library is required but not installed.
   Install with: pip install deap
   ```

2. **GPU関連のエラー**
   - オリジナル版と同様にCUDA/CPUの自動選択を行います

## 開発者向け情報

### カスタムオペレータの追加

DEAPの豊富なオペレータライブラリを活用できます：

```python
# 新しい選択方法を試す
toolbox.register("select", tools.selRoulette)  # ルーレット選択
toolbox.register("select", tools.selNSGA2)    # NSGA-II選択

# 新しい交叉方法を試す
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # 一様交叉
toolbox.register("mate", tools.cxOrdered)              # 順序交叉
```

### 統計情報の拡張

```python
# カスタム統計を追加
stats.register("diversity", tools.diversity, first=operator.attrgetter("fitness.values"))
```

## まとめ

DEAP版は、より標準的で拡張しやすいGA実装を提供しながら、元の機能をすべて保持しています。研究目的や、様々なGA戦略を試したい場合に特に有用です。

