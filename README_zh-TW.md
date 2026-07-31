# FAD · Flow-Aligned Distillation 研究作品集

[線上作品集](https://hanklin9188.github.io/FAD-Portfolio/) · [完整數據](docs/RESULTS.md) · [復現指南](reproduction/README.md) · [25% 部署指南](deployment/README.md)

這個 repo 將 FAD 的研究成果整理成可閱讀、可稽核、可復現、可展示的作品集。內容涵蓋 Llama-3.2-3B 的 15%、20%、25%、30% 結構壓縮、Adaptive Exit、三個論文方法的比較，以及 25% student 與 merged teacher 的網站部署範例。

## 最新選定結果

數字是 PIQA、Social-IQA、WinoGrande、ARC-Challenge、ARC-Easy、HellaSwag、OpenBookQA 七項 zero-shot accuracy 的未加權平均。

| 壓縮率 | 獨立 FFN 數 | Static FAD | Adaptive Exit | 差異 | 平均出口層 | 省略層數比例 |
|---:|---:|---:|---:|---:|---:|---:|
| 15% | 22 / 28 | **85.38%** | **85.41%** | +0.04 pp | 17.32 | 38.16% |
| 20% | 19 / 28 | **85.66%** | **85.53%** | −0.14 pp | 18.16 | 35.16% |
| 25% | 17 / 28 | **85.03%** | **84.94%** | −0.10 pp | 16.39 | 41.47% |
| 30% | 15 / 28 | **83.20%** | **82.89%** | −0.31 pp | 17.16 | 38.70% |

25% 模型將 28 層對應到 17 組獨立 FFN：第 0–13、27 層各自保留；第 14–18 層共用一組，第 19–26 層共用另一組。這等同減少 11 組 FFN，約 0.83B 參數或原始 3.21B 模型的 25.8%。

## 重要的數據稽核

- 論文草稿把 `1.463×` 稱為 task-wise geometric mean；來源 JSON 顯示其實 **1.465× 是 aggregate throughput ratio**，真正的七任務 geometric mean 是 **1.388×**。作品集已分開標示。
- 最新 FAD JSON 明確使用 `length_norm=none`；先前 baseline wrapper 預設為 `avg`，而舊 JSON 沒有保存此欄位。因此目前比較表保留論文時期結果，但嚴格的同協定結論仍需要全部重跑。公開腳本已統一成 `none`。
- 19,149 題配對測試中，Adaptive student 答對 17,079 題，teacher 答對 16,726 題；student-only 正確 1,189 題、teacher-only 正確 836 題，淨差 +353。這是選擇題結果，不能直接外推到自由生成聊天。

## 怎麼使用

- 想看每個檔案的用途：[`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md)
- 想看每個 task 與 paper 比較：[`docs/RESULTS.md`](docs/RESULTS.md)
- 想重跑四個壓縮率與 Adaptive Exit：[`reproduction/README.md`](reproduction/README.md)
- 想部署 25% student + teacher：[`deployment/README.md`](deployment/README.md)
- 想查實驗公平性與已知限制：[`docs/AUDIT.md`](docs/AUDIT.md)

模型權重、完整 benchmark 與大型 paired-question 檔沒有公開；mock 展示介面保留 10 題 HellaSwag 範例、答案與歷史輸出，且清楚標示不是即時推論。部署所需三個 artifact 的 SHA-256 已寫在 `deployment/web-demo/models/README.md`。
