# Flow-Aligned Distillation

這是 ICLR 2027 論文〈Flow-Aligned Distillation: Preserving Layer-Wise Residual Trajectories in Compressed Language Models〉的公開、無權重研究 artifact。

FAD 先以跨層共享 FFN bank 壓縮 Transformer，再於 teacher-derived 低秩座標中對齊 teacher 與 student 的 FFN residual update。公開版依照「論文 claim → 精簡結果表 → 分析/產圖程式 → 排程實驗腳本」整理，並刻意不包含 checkpoints、模型權重、geometry tensors、完整資料集、憑證與 Slurm logs。

## 主結果

主要實驗固定 Llama-3.2-3B、`K=19` FFN banks、20.93% whole-model parameter reduction、teacher、資料、初始化、optimizer、checkpoint rule 與 evaluator，只改 recovery objective。三個獨立 training seeds（42/43/44）的結果如下：

| Recovery objective | Commonsense | MMLU 0-shot | MMLU 5-shot |
|---|---:|---:|---:|
| CE only | 85.10 ± 0.57 | 33.37 ± 5.34 | 34.08 ± 4.47 |
| Logit KL | 84.91 ± 0.08 | 39.88 ± 1.12 | 43.38 ± 0.47 |
| Hidden-state MSE | 85.06 ± 0.07 | 44.26 ± 2.21 | 45.71 ± 2.31 |
| Ambient velocity MSE | **85.17 ± 0.39** | 40.99 ± 1.47 | 43.50 ± 1.42 |
| Projected isotropic velocity | 84.96 ± 0.06 | 45.02 ± 0.72 | 48.60 ± 1.12 |
| **FAD** | 84.84 ± 0.42 | **47.55 ± 3.84** | **52.09 ± 1.96** |

FAD 相較最強的 non-FAD 平均值，在 MMLU 0-shot / 5-shot 分別提升 2.53 / 3.49 個百分點，且三個 seeds 都勝過 projected isotropic matching。

## 快速導覽

- [完整論文原始碼與 16 張引用圖](paper/README.md)
- [每個實驗數據、圖與程式的對照索引](docs/EXPERIMENT_INDEX.md)
- [完整 reproduction 說明](reproduction/README.md)
- [公開 primary manifest](experiments/iclr2027/provenance/primary_k19_manifest.json)

無 GPU 的一致性驗證：

```bash
python scripts/verify_data.py
python scripts/verify_paper_release.py
```

重建 13 張由公開表格產生的論文圖：

```bash
python scripts/generate_paper_figures.py
```

在叢集環境請一律透過 scheduler 提交，不可在 login node 執行 Python、產圖、編譯或模型工作。

## Provenance 修正

公開 manifest 已分開兩個曾在內部草稿混淆的 rank：teacher fine-tuning LoRA 是 64/64；壓縮 student 的 private adapter 是 128/128。公開論文原始碼也已同步為 audited 值。

模型、資料集與第三方專案仍受各自授權條款約束，詳見 [THIRD_PARTY.md](THIRD_PARTY.md)。
