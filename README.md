# talksplit

> [English version](README.en.md)

用句子嵌入加 TextTiling 深度評分，把講稿、逐字稿、文章切成語意連貫的段落。
中文、英文、中英混合都可以。**不限長度**。

兩種使用方式：
- **CLI** — `talksplit input.txt`
- **ComfyUI** — 整個 repo 直接放進 `custom_nodes/`

## 安裝

```bash
python3 INSTALL.py
```

安裝器會自動偵測：
- **目前已啟用 venv / conda env**（例如你從 ComfyUI 的 venv 內執行）→ 直接裝在那裡，不另外建 `.venv`
- **系統 Python** → 在 repo 內建立 `.venv` 並裝在裡面

預設模型是 `BAAI/bge-m3`（約 2 GB）。常見變化：

```bash
python3 INSTALL.py --model intfloat/multilingual-e5-small   # 約 470 MB
python3 INSTALL.py --skip-model                             # 只裝套件，第一次跑時再下載模型
python3 INSTALL.py --skip-venv                              # 用當前 Python（如 ComfyUI portable）
python3 INSTALL.py --force-venv                             # 即使已啟用 venv 也另建本地 .venv
```

## CLI 用法

```bash
# 基本用法 —— 預設 "speech" 模式（sensitivity 1.0、min 2、max 15）
talksplit input.txt

# 文章 / 部落格模式（sensitivity 0.7、min 4、max 30）
talksplit input.txt --mode article

# 調整切段嚴格度（0–2，越高切越多）—— 會覆蓋 mode 預設
talksplit input.txt --sensitivity 1.4

# 強制剛好 8 段
talksplit input.txt --target 8

# 限制段落長度（句子數）
talksplit input.txt --min-sentences 3 --max-sentences 12

# JSON 輸出，含每段的句子索引
talksplit input.txt --format json

# 除錯：輸出相似度 / depth 曲線圖
talksplit input.txt --plot curve.png

# 換模型
talksplit input.txt --model intfloat/multilingual-e5-small

# 允許切點落在逗號（不只是句末）
talksplit input.txt --clause-level

# 關閉轉折詞加成（如「Let me switch」「接下來」）
talksplit input.txt --no-markers
```

### 兩種模式

| 模式 | sensitivity | min_sentences | max_sentences | 適用 |
|---|---|---|---|---|
| `speech`（預設） | 1.0 | 2 | 15 | 講稿、演講逐字稿 |
| `article` | 0.7 | 4 | 30 | 部落格文章、長篇論述 |

明確指定 `--sensitivity` / `--min-sentences` / `--max-sentences` 會覆蓋 mode 的預設值。

## ComfyUI 安裝

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> talksplit
cd talksplit

# 方式一 —— ComfyUI Manager 會自動處理 requirements.txt
# 方式二 —— 手動：從 ComfyUI 的 venv 裡跑我們的安裝器
#         （會自動偵測啟用中的 venv，不會另建）
python INSTALL.py
# 方式三 —— Windows ComfyUI portable：
../../python_embeded/python.exe INSTALL.py --skip-venv
```

重啟 ComfyUI，節點會出現在 **`talksplit`** 分類下。

> **資料夾名稱要小心**：clone 時請命名為 `talksplit/` 或任何**不是 `split/`** 的名字。
> bridge 已盡量處理碰撞，但 `split` 會與內部 Python package 衝突。

## ComfyUI 節點

一鍵：
- **Talksplit · Auto** — 文字進、段落出，內含整條管線

細粒度（給自訂工作流）：
- **Talksplit · Sentences** — 句切分（含無標點 fallback）
- **Talksplit · Embed** — 透過 `sentence-transformers` 算句嵌入
- **Talksplit · Score** — 相鄰餘弦相似度 + TextTiling depth
- **Talksplit · Marker Boost** — 在轉折詞前的 gap 加深 depth
- **Talksplit · Optimize** — 受長度約束的 DP 選切點
- **Talksplit · Assemble** — 依切點把句子組成段落
- **Talksplit · Plot** — 繪製相似度 / depth 曲線為 `IMAGE`

下游接管用：
- **Talksplit · Split to List** — 把 `paragraphs` STRING 拆成 list（標記
  `OUTPUT_IS_LIST=True`），下游每段跑一次。例如 `Auto → Split to List →
  CLIPTextEncode → KSampler → SaveImage` 會替每段各產一張圖。
- **Talksplit · Pick Paragraph** — 用 `index` 取出第 N 段，輸出 `paragraph` 和
  總段數 `count`。負索引（-1 = 最後一段）和越界自動 clamp。

`workflows/` 內附兩個範例：
- `basic.json` — 只用一顆 Auto 節點
- `pipeline.json` — 完整細粒度管線含除錯圖

## 運作原理

1. **句切分**：以標點為主訊號，長段無標點才走 fallback 視窗切分。
2. **嵌入**：每句一個向量（預設 BGE-M3，多語言）。
3. **邊界打分**：相鄰句的餘弦相似度 → TextTiling 風格的 depth 分數。
   depth 反映的是「相對於兩側峰值，這個低谷有多深」，不只是看絕對值低不低。
4. **閾值**：`mean(depth) + k·std(depth)`，`k` 由 `sensitivity`（0–2）推導。
   sensitivity 1.0 只取深谷；2.0 連 mean 以上都收。
5. **轉折詞加成**（預設開啟，`--no-markers` 關閉）：句首匹配「Let me switch」、
   「接下來」這類轉折詞時，前面那個 gap 加 depth bonus。閾值用「加成前」的 depth
   分布計算，所以單一 marker 加成不會拉高閾值反過來把自然深谷擠下去。
6. **長度約束 DP**：在每段 `[min_sentences, max_sentences]` 範圍內，挑出讓 depth
   總和最大的切點集。如果閾值濾掉所有可行切點，候選集會自動放寬。
7. **可選 `--target N`**：忽略閾值，強制挑出 `N − 1` 個切點，總 depth 最大化。
8. **可選 `--clause-level`**：也把逗號當終止符，DP 可以在句子中段挑點切。
   對「逗號串長句」（口語逐字稿常見）有用。

標點只是**訊號**（決定句子在哪結束），段落不是用標點切的，是用語意凝聚力決定的。

## 測試

```bash
.venv/bin/pytest tests/ -v
```

## 專案結構

```
__init__.py            ComfyUI 載入點（用 importlib 把內部 package 改名）
INSTALL.py             CLI 模式一鍵安裝器
pyproject.toml         CLI 模式的 Python packaging
requirements.txt       ComfyUI 自動安裝清單
src/split/
  segmenter.py         句切分
  embedder.py          嵌入模型封裝
  boundary.py          相似度 + TextTiling depth
  markers.py           轉折詞表 + 加成函式
  optimizer.py         長度約束 DP
  visualize.py         CLI 曲線圖
  comfy_nodes.py       ComfyUI 節點類別
  cli.py               CLI 入口
workflows/
  basic.json           單顆 Auto 節點
  pipeline.json        完整管線
tests/                 單元測試
```
