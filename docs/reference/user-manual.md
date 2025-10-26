# GOAP Regression Pipeline User Manual / 利用マニュアル

## English

### Overview
The GOAP (Goal-Oriented Action Planning) Regression Pipeline automates an end-to-end regression workflow. It ingests a CSV dataset, prepares the data, trains a regression model, and reports evaluation metrics such as R-squared (R2), root mean squared error (RMSE), and mean absolute error (MAE). Execution is orchestrated by a GOAP planner that selects and runs the required actions until the evaluation goal is achieved.

### Prerequisites
- Python environment prepared via [`uv`](https://github.com/astral-sh/uv).
- All project dependencies installed with `uv sync --extra dev` from the repository root.
- Source CSV file containing the regression target column you want to predict.

### Running the Pipeline
1. Open a shell in the repository root.
2. Ensure dependencies are installed (`uv sync --extra dev`).
3. Execute the CLI with the `goapml run` command, for example:

   ```bash
   goapml run \
     --csv ./data/house-prices.csv \
     --target SalePrice \
     --model random_forest \
     --test-size 0.2 \
     --metrics r2 rmse mae \
     --json-out results/run.json \
     --log-level INFO
   ```

The command prints a JSON payload to stdout summarising the run. If `--json-out` is supplied, the same payload is written to the specified file.

### Command Options
| Option | Required | Description |
| --- | --- | --- |
| `--csv PATH` | ✅ | Path to the input CSV file (must exist and be readable). |
| `--target TEXT` | ✅ | Name of the regression target column. |
| `--encoding TEXT` | ❌ | Explicit text encoding for the CSV. Leave empty to auto-detect. |
| `--delimiter TEXT` | ❌ | Field delimiter (default: `,`). |
| `--decimal TEXT` | ❌ | Decimal separator (default: `.`). |
| `--header / --no-header` | ❌ | Declare whether the CSV includes a header row (default: header present). |
| `--model [linear_regression|random_forest|xgboost|...]` | ❌ | Regression algorithm to train (default: linear regression). |
| `--test-size FLOAT` | ❌ | Validation split fraction between 0.05 and 0.5 (default: 0.2). |
| `--metrics METRIC ...` | ❌ | One or more evaluation metrics (`r2`, `rmse`, `mae`). When omitted the default trio is used. |
| `--json-out PATH` | ❌ | Optional path for saving the JSON payload. Symlinked directories are rejected for safety. |
| `--log-level [CRITICAL|ERROR|WARNING|INFO|DEBUG]` | ❌ | Logging verbosity for the run (default: INFO). |

Additional metric names can also be passed as positional arguments after the options block. Unknown names trigger a validation error.

### Output
- **Stdout**: Structured JSON containing run status (`ok` or `error`), executed actions, metrics, and accumulated logs.
- **JSON file (optional)**: Same payload as stdout. The CLI refuses to write to paths outside the current working directory or into symlinked directories.
- **Exit codes**: `0` for success, `1` if configuration or execution fails.

### Troubleshooting
- Ensure the CSV path is correct and accessible.
- Verify the target column exists and contains numeric data suitable for regression.
- Check that the test split (`--test-size`) stays within the allowed range.
- If execution stops early, inspect the `logs` array in the JSON output to see which GOAP actions ran before failure.

### Further Reading
- [Project README](../../README.md)
- [GOAP Regression Pipeline Plan](plan-and-tasks.md)

---

## 日本語

### 概要
GOAP（Goal-Oriented Action Planning）回帰パイプラインは、CSV データの取り込みから前処理、回帰モデルの学習、評価指標の算出までを自動化します。GOAP プランナーが必要なアクションを選択・実行し、評価ゴールを達成するまで処理を進めます。出力では決定係数（R2）、二乗平均平方根誤差（RMSE）、平均絶対誤差（MAE）などの指標を確認できます。

### 前提条件
- [`uv`](https://github.com/astral-sh/uv) で管理された Python 環境。
- リポジトリ直下で `uv sync --extra dev` を実行し依存関係をインストール済みであること。
- 予測したい目的変数（ターゲット列）を含む CSV ファイル。

### 実行手順
1. リポジトリのルートディレクトリでシェルを開きます。
2. 依存パッケージをインストール済みか確認します（`uv sync --extra dev`）。
3. `goapml run` コマンドを実行します。例：

   ```bash
   goapml run \
     --csv ./data/house-prices.csv \
     --target SalePrice \
     --model random_forest \
     --test-size 0.2 \
     --metrics r2 rmse mae \
     --json-out results/run.json \
     --log-level INFO
   ```

コマンドの実行結果は JSON として標準出力に表示されます。`--json-out` を指定すると同じ内容がファイルにも保存されます。

### コマンドオプション
| オプション | 必須 | 説明 |
| --- | --- | --- |
| `--csv PATH` | ✅ | 入力となる CSV ファイルへのパス（存在し読み取り可能である必要があります）。 |
| `--target TEXT` | ✅ | 予測対象とする目的変数（列名）。 |
| `--encoding TEXT` | ❌ | CSV の文字エンコーディング。空欄で自動判定。 |
| `--delimiter TEXT` | ❌ | CSV の区切り文字（デフォルトは `,`）。 |
| `--decimal TEXT` | ❌ | 小数点の区切り文字（デフォルトは `.`）。 |
| `--header / --no-header` | ❌ | CSV にヘッダー行が含まれるかどうか（デフォルトはヘッダーあり）。 |
| `--model [linear_regression|random_forest|xgboost|...]` | ❌ | 使用する回帰アルゴリズム（デフォルトは線形回帰）。 |
| `--test-size FLOAT` | ❌ | 検証用データの割合。0.05〜0.5 の範囲で指定（デフォルトは 0.2）。 |
| `--metrics METRIC ...` | ❌ | 評価指標（`r2`、`rmse`、`mae` など）を複数指定可能。省略時は標準の 3 指標を計算します。 |
| `--json-out PATH` | ❌ | JSON 出力を保存するファイルパス。シンボリックリンクを含むディレクトリは安全のため拒否されます。 |
| `--log-level [CRITICAL|ERROR|WARNING|INFO|DEBUG]` | ❌ | ログの詳細度（デフォルトは INFO）。 |

オプションで指定しなかった評価指標は、オプションの後に位置引数として追加できます。未知の名前を渡すとバリデーションエラーになります。

### 出力内容
- **標準出力**: 実行結果の JSON。ステータス（`ok` / `error`）、実行されたアクション、評価指標、ログが含まれます。
- **JSON ファイル（任意）**: 標準出力と同一内容。現在の作業ディレクトリ外やシンボリックリンクを含むディレクトリには書き込みません。
- **終了コード**: 成功時は `0`、設定エラーや実行エラーが発生した場合は `1` を返します。

### トラブルシューティング
- CSV のパスが正しいか、アクセス権限があるか確認してください。
- ターゲット列が存在し、回帰に適した数値データであることを確かめてください。
- テストデータの割合（`--test-size`）が許容範囲内か確認してください。
- 途中で処理が停止した場合は、JSON の `logs` 配列から GOAP アクションの実行状況を確認してください。

### 参考資料
- [プロジェクト README](../../README.md)
- [GOAP 回帰パイプライン計画書](plan-and-tasks.md)

