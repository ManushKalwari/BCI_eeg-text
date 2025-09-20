# EEG → Text (BCI_eeg-text)

**Modular EEG-to-Text decoding** that:
1) self-supervises an EEG encoder,  
2) aligns EEG ↔ text embeddings via a CLIP-style contrastive loss, and  
3) decodes with a BART-based autoregressive head.

> TL;DR: We align noisy EEG to language embeddings, then generate text with a pretrained LM. The pipeline is designed for **generalization without teacher forcing** at inference. 
---

## 🚀 Key Ideas
- **EEG encoder pretraining** with masked autoencoding to learn local/global temporal structure. 
- **Contrastive alignment**: project EEG and BERT embeddings to a shared space; optimize symmetric InfoNCE with temperature.  
- **Decoder**: freeze the EEG encoder, condition a **BART** decoder on the projected EEG vector; train autoregressively. 
- **No teacher forcing at inference** (to mirror real deployments). 

---

## 🧠 Dataset: ZuCo 2.0 (Natural Reading)
We use **ZuCo 2.0** (EEG during natural reading). EEG sampled with 128-channel BioSemi at 500 Hz; we use the NR subset. Download per the dataset license and place under `data/`. 

Preprocessing (high level):
- segment time series to overlapping patches;  
- artifact removal + normalization;  
- align word-level EEG segments with text;  
- produce word/phrase text labels.

---

## 🛠️ Setup
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
# expects: torch, transformers, datasets, numpy, scipy, scikit-learn, einops, tqdm, matplotlib, wandb (optional)
