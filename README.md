# EEG → Text (BCI_eeg-text)

**Modular EEG-to-Text decoding** that:
1) self-supervises an EEG encoder,  
2) aligns EEG ↔ text embeddings via a CLIP-style contrastive loss, and  
3) decodes with a BART-based autoregressive head. :contentReference[oaicite:1]{index=1}

> TL;DR: We align noisy EEG to language embeddings, then generate text with a pretrained LM. The pipeline is designed for **generalization without teacher forcing** at inference. :contentReference[oaicite:2]{index=2}

---

## 🚀 Key Ideas
- **EEG encoder pretraining** with masked autoencoding to learn local/global temporal structure. :contentReference[oaicite:3]{index=3}  
- **Contrastive alignment**: project EEG and BERT embeddings to a shared space; optimize symmetric InfoNCE with temperature. :contentReference[oaicite:4]{index=4}  
- **Decoder**: freeze the EEG encoder, condition a **BART** decoder on the projected EEG vector; train autoregressively. :contentReference[oaicite:5]{index=5}  
- **No teacher forcing at inference** (to mirror real deployments). :contentReference[oaicite:6]{index=6}

---

## 📦 Repo Structure
BCI_eeg-text/
├─ data/ # (expected) ZuCo 2.0 processed tensors / metadata
├─ eeg_pretrain/ # masked autoencoder for EEG
├─ align/ # EEG↔Text contrastive alignment (CLIP-style)
├─ decode/ # BART-based autoregressive decoder
├─ scripts/ # end-to-end run scripts
├─ utils/ # common loaders, metrics, logging
└─ README.md


_If your file layout differs, adjust these paths in the commands below._

---

## 🧠 Dataset: ZuCo 2.0 (Natural Reading)
We use **ZuCo 2.0** (EEG during natural reading). EEG sampled with 128-channel BioSemi at 500 Hz; we use the NR subset. Download per the dataset license and place under `data/`. :contentReference[oaicite:7]{index=7}

Preprocessing (high level):
- segment time series to overlapping patches;  
- artifact removal + normalization;  
- align word-level EEG segments with text;  
- produce word/phrase text labels. :contentReference[oaicite:8]{index=8}

---

## 🛠️ Setup
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
# expects: torch, transformers, datasets, numpy, scipy, scikit-learn, einops, tqdm, matplotlib, wandb (optional)
