# FAD · Flow-Aligned Distillation 研究作品集

[英文首頁](README.md) · [完整數據](docs/RESULTS.md) · [復現指南](REPRODUCIBILITY.md) · [實驗腳本](reproduction/README.md) · [部署指南](deployment/README.md) · [限制與適用範圍](LIMITATIONS.md)

FAD 是一套針對 Llama-3.2-3B 的跨層 FFN 權重共用與功能恢復方法。它將 28 個 decoder layer 對應到較少的獨立 FFN 權重，再利用每層私有的低秩 adapter，對齊 teacher 定義的 projected FFN residual update。推論階段另有一個四特徵 controller，可在第 16、20、24 或 28 層提前結束。

這個 repository 的重點不只是保存程式碼，而是提供一條可稽核的證據鏈：

```text
原始 evaluator / controller artifact
→ processed CSV
→ scripts/verify_data.py
→ README 與結果文件
```

## 最新選定結果

數字是 PIQA、Social-IQA、WinoGrande、ARC-Challenge、ARC-Easy、HellaSwag、OpenBookQA 七項 zero-shot accuracy 的未加權平均。

| 壓縮率 | 獨立 FFN 數 | Static FAD | Adaptive Exit | 差異 | 平均出口層 | 省略層數比例 |
|---:|---:|---:|---:|---:|---:|---:|
| 15% | 22 / 28 | **85.38%** | **85.41%** | +0.04 pp | 17.32 | 38.16% |
| 20% | 19 / 28 | **85.66%** | **85.53%** | −0.14 pp | 18.16 | 35.16% |
| 25% | 17 / 28 | **85.03%** | **84.94%** | −0.10 pp | 16.39 | 41.47% |
| 30% | 15 / 28 | **83.20%** | **82.89%** | −0.31 pp | 17.16 | 38.70% |

25% 的 archived H100 batch-1 runtime decomposition 需要區分兩個不同統計量：

- **1.465× aggregate throughput ratio**。
- **1.388× 七任務 task-wise geometric-mean speedup**。

兩者不能互換，`scripts/verify_data.py` 會從 retained artifacts 重新計算。

## 協定狀態

目前 FAD artifact 明確保存 `length_norm=none`。歷史 FLAP、Týr-the-Pruner 與 LLM-Streamline 結果則來自 paper-era workflow，當時 wrapper 的預設 normalization 與目前協定不完全相同。因此相關表格保留作為歷史比較背景，但在所有方法以同一個 frozen protocol 重跑前，不宣稱它們是完整的 apples-to-apples comparison。

請先閱讀 [`docs/AUDIT.md`](docs/AUDIT.md) 與 [`LIMITATIONS.md`](LIMITATIONS.md) 再引用跨方法結論。

## 不需要 GPU 的驗證

```bash
python scripts/verify_data.py
```

這個 verifier 會：

- 從 raw artifacts 重算 macro accuracy；
- 重算 aggregate 與 task-wise runtime summary；
- 檢查 adaptive-exit 與 paired-question accounting；
- 檢查網站內部連結；
- 掃描常見 private path marker。

也可以啟動不含模型權重的 mock deployment：

```bash
bash deployment/web-demo/start-mock.sh 8765 &
server_pid=$!
python deployment/web-demo/smoke_test.py --base_url http://127.0.0.1:8765
kill "$server_pid"
```

Mock mode 只驗證 API 與介面 contract，不代表即時模型推論，也不能支持效能或模型品質結論。

## 完整實驗

完整 GPU reproduction 需要另外取得模型權重與資料集，並依 [`reproduction/README.md`](reproduction/README.md) 設定路徑。叢集實驗應透過 Slurm 提交，不在 login node 執行 GPU 或重型 CPU 工作：

```bash
sbatch reproduction/fad/slurm/fad_budgets.sbatch
```

## 目前狀態

| 項目 | 狀態 |
|---|---|
| 公開數據一致性檢查 | **可執行且由 CI 強制** |
| Weight-free deployment smoke test | **可執行且由 CI 強制** |
| Selected FAD budget artifacts | **已公開** |
| 歷史 baseline context | **已公開，附協定限制** |
| 完整 matched baseline rerun | **尚未完成** |
| 模型權重與完整資料集 | **刻意不重新散布** |

公開文件不宣稱特定投稿、審查、錄取或 camera-ready 狀態。模型權重、完整 benchmark、大型 per-question records 與 deployable student bundle 也不包含在 repository 中。
