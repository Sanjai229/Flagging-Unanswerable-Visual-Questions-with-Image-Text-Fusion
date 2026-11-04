# Flagging Unanswerable Visual Questions with Image-Text Fusion
 
Individual submission for Project 2.2, Multimodal Data Fusion, Adelaide University, Semester 2 2026.
 
Blind and low-vision people photograph objects and ask questions about them, but many of those questions cannot be answered from the photo because it is blurred, badly framed or misses the object (Gurari et al. 2018). This project predicts whether a question can be answered from the **photo and the question together**, and compares single-modality baselines with late, early, gated and cross-attention fusion.
 
[![Open notebook 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanjai229/Flagging-Unanswerable-Visual-Questions-with-Image-Text-Fusion/blob/main/notebooks/02_train_and_evaluate.ipynb)
 
## Task and data
 
- **Data:** VizWiz-VQA, CC BY 4.0 (VizWiz n.d.). Official splits: train (20,523 pairs) for training, validation (4,319) as **dev** for early stopping, thresholds and the late-fusion weight, and test (8,000, labels from the VizWiz all-answers release) for final scores only.
- **Label:** 1 = unanswerable (about 27% of pairs). Positive class for every metric.
- **Inputs:** frozen ViT-B/16 photo features, frozen BERT-base question features, and CLIP features as a reference. The crowd answers are used only to build labels and for analysis, never as model inputs.
## Models
 
| Name | Group | Idea |
|---|---|---|
| `majority`, `photo_quality` | floor | constant prediction; logistic regression on photo sharpness, brightness and size |
| `question_only`, `image_only` | main | MLP on one modality |
| `late` | main | weighted average of the two single-modality probabilities, weight tuned on dev |
| `early` | main | MLP on the joined BERT and ViT vectors |
| `gated` | main | Gated Multimodal Unit (Arevalo et al. 2017) with modality dropout |
| `cross_attention` | main | bidirectional cross-attention between BERT word tokens and ViT region tokens (Vaswani et al. 2017; Tsai et al. 2019) |
| `*_soft`, `gated_no_modality_dropout`, `cross_attention_2layers` | ablation | one change each |
| `clip_early`, `clip_gated` | reference | the same fusion with CLIP encoders (Radford et al. 2021) |
 
Every trainable model uses the same loop: BCE loss with `pos_weight` for the class imbalance, AdamW (Loshchilov & Hutter 2019), early stopping on dev AUPRC, 3 seeds. All settings are in [`configs/default.yaml`](configs/default.yaml).
 
## Results
 
Official test set (8,000 pairs), mean of 3 seeds, decision thresholds tuned on dev only.
 
| Model | Macro-F1 [95% CI] | AUROC | AUPRC |
|---|---|---|---|
| Majority class | 0.423 [0.419, 0.426] | 0.500 | 0.268 |
| Photo quality only | 0.613 [0.602, 0.625] | 0.660 | 0.453 |
| Question only | 0.663 [0.653, 0.672] | 0.748 | 0.536 |
| Image only | 0.694 [0.684, 0.704] | 0.778 | 0.597 |
| Late fusion | 0.742 [0.732, 0.752] | 0.842 | 0.674 |
| **Early fusion** | **0.771 [0.762, 0.781]** | **0.871** | **0.733** |
| Gated fusion | 0.756 [0.746, 0.765] | 0.858 | 0.708 |
| Cross-attention | 0.766 [0.756, 0.776] | 0.867 | 0.719 |
| CLIP early fusion (reference) | 0.800 [0.791, 0.809] | 0.900 | 0.777 |
 
Every fusion model beats both single-modality models. Early fusion is best: +0.077 macro-F1 over image only (paired bootstrap 95% CI 0.068 to 0.086), at 37.5 ms per query on a T4. Cross-attention and gated fusion did not beat it, while better-aligned encoders (CLIP) did. All models score lower on sharp photos than on blurry ones and on rare questions than on common ones; details are in the report.
 
The evaluation notebook saves every figure and table to [`results/`](results/): test metrics (mean +/- sd over seeds and bootstrap 95% CIs), paired gains, learning curves, cost, ablations, missing-input tests, subgroup results (photo sharpness, question frequency, answer type), calibration, operating points and failure cases.
 
## Reproduce
 
Everything runs on a free Google Colab T4 GPU with Google Drive for storage.
 
1. **Features (about 45 minutes the first time):** open [`notebooks/01_data_and_features.ipynb`](notebooks/01_data_and_features.ipynb) in Colab, choose Runtime > Change runtime type > T4 GPU, then Runtime > Run all and allow Drive access. It downloads the data (about 20 GB of photos, kept on the Colab disk only) and saves features, `meta.csv` and `manifest.json` to `MyDrive/vizwiz_fusion/features/` (about 4.7 GB).
2. **Training and evaluation (about 20 minutes, of which training takes 9):** open [`notebooks/02_train_and_evaluate.ipynb`](notebooks/02_train_and_evaluate.ipynb) the same way and Run all. It clones this repository, runs the smoke test, trains every model with `python -m vizwiz.train`, and evaluates them with `vizwiz/evaluate.py`. Runs go to `MyDrive/vizwiz_fusion/individual_runs/`, figures and tables to `MyDrive/vizwiz_fusion/results_for_github/`.
Command line (from the repo root, after notebook 1):
 
```bash
pip install -r requirements.txt
python -m tests.test_smoke                       # about a minute on CPU, synthetic data
python -m vizwiz.train --config configs/default.yaml --models all
python -m vizwiz.train --models gated cross_attention --seeds 0   # a subset
```
 
## Repository structure
 
```
configs/default.yaml                 every setting: paths, training, models and hyperparameters
vizwiz/data.py                       loads features and meta.csv, splits, batches, soft labels
vizwiz/models.py                     MLP (single modality, early fusion), cross-attention, gated fusion
vizwiz/train.py                      shared training loop, late fusion, floor models, command line
vizwiz/evaluate.py                   metrics, bootstrap CIs, overfitting, subgroups, robustness, failure cases
notebooks/01_data_and_features.ipynb data download, labels, data quality, feature extraction, encoder latency
notebooks/02_train_and_evaluate.ipynb training and full evaluation with figures
tests/test_smoke.py                  end-to-end check on synthetic data
results/                             figures and aggregate tables from the evaluation notebook
```
 
## Ethics
 
Only code, figures and aggregate results are shared here: no photos, questions or annotations are redistributed, and `meta.csv` and the features stay on the author's Drive. Example photos in the report are attributed to the VizWiz dataset. The bias assessment (photo quality, question frequency, answer type) and deployment considerations are in the report.
 
## Licence
 
Code: MIT (see `LICENSE`). Data: CC BY 4.0, owned by the VizWiz team.
 
## References
 
Arevalo, J, Solorio, T, Montes-y-Gomez, M & Gonzalez, FA 2017, 'Gated multimodal units for information fusion', in *International conference on learning representations (ICLR 2017), workshop track*, Toulon, France.
 
Gurari, D, Li, Q, Stangl, AJ, Guo, A, Lin, C, Grauman, K, Luo, J & Bigham, JP 2018, 'VizWiz grand challenge: answering visual questions from blind people', in *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, IEEE, Salt Lake City, USA, pp. 3608-3617.
 
Loshchilov, I & Hutter, F 2019, 'Decoupled weight decay regularization', in *International conference on learning representations (ICLR 2019)*, New Orleans, USA.
 
Radford, A, Kim, JW, Hallacy, C, Ramesh, A, Goh, G, Agarwal, S, Sastry, G, Askell, A, Mishkin, P, Clark, J, Krueger, G & Sutskever, I 2021, 'Learning transferable visual models from natural language supervision', in *Proceedings of the 38th international conference on machine learning (ICML 2021)*, PMLR, vol. 139, pp. 8748-8763.
 
Tsai, Y-HH, Bai, S, Liang, PP, Kolter, JZ, Morency, L-P & Salakhutdinov, R 2019, 'Multimodal transformer for unaligned multimodal language sequences', in *Proceedings of the 57th annual meeting of the Association for Computational Linguistics*, Association for Computational Linguistics, Florence, Italy, pp. 6558-6569.
 
Vaswani, A, Shazeer, N, Parmar, N, Uszkoreit, J, Jones, L, Gomez, AN, Kaiser, L & Polosukhin, I 2017, 'Attention is all you need', in *Advances in neural information processing systems 30 (NeurIPS 2017)*, Curran Associates, Long Beach, USA, pp. 5998-6008.
 
VizWiz n.d., *Visual question answering*, VizWiz, viewed 4 November 2026, <https://vizwiz.org/tasks-and-datasets/vqa/>.
