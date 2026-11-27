# 🌐 Multilingual Intent Classification & Out-of-Scope Detection
**Top 15 Finalist — Anveshan Data Sprint 2.0 (NLP Track) | The National Institute of Engineering (NIE), Mysuru**

**Developed by [Krish Nayyar](https://github.com/krishnayyar999-sudo) (`Team: krishnayyar999`)** | [LinkedIn](https://www.linkedin.com/in/krish-nayyar-071240420)

[![Result](https://img.shields.io/badge/Result-Top%2015%20Finalist-gold)](https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Executed%20Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07)
[![Macro-F1](https://img.shields.io/badge/Public%20LB%20Macro--F1-0.77858-brightgreen)](https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)

---

## 🔗 Quick Links & Competition Highlights
- **Achievement:** **Top 15 Finalist** — Anveshan Data Sprint 2.0 (The National Institute of Engineering, Mysuru)
- **Live Kaggle Notebook (With Executed Outputs):** [View Notebook on Kaggle](https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07)
- **Best Public Leaderboard Score:** **`0.77858`** (Macro-F1 across 56 classes)
- **Team Name:** `krishnayyar999`
- **Official Finalist Certificate:** *Pending issuance from organizers (Event concluded Nov 2026)*

---

## 📌 Problem Statement
Conversational AI assistants operating across multilingual and code-mixed regions face two core challenges:
1. **Fine-Grained Intent Routing:** Accurately classifying noisy, short, transliterated, and code-mixed user queries across **55 distinct intent categories** (`Classes 0–54`).
2. **Out-of-Scope (OOS) Detection:** Reliably identifying unsupported, adversarial, or irrelevant queries (**Class 55**) so the system falls back gracefully instead of triggering false downstream actions.

Because real-world intent distributions are heavily skewed, standard cross-entropy accuracy models overfit to majority intents and collapse on rare tail classes. This pipeline directly optimizes **Macro-F1** across all **56 classes** (`0–55`).

---

## 🧠 Solution Architecture
To capture both surface-level transliteration patterns and deep cross-lingual semantics, the pipeline blends **three complementary model families** followed by metric-aligned probability calibration:

1. **Input Preprocessing & Normalization:** Unicode normalization, whitespace/punctuation cleanup, and code-mixed token preservation on raw multilingual queries.
2. **Parallel 3-Branch Ensemble:**
   * **Branch 1 — Statistical Lexical Model:** Word N-grams `(1–3)` + Character word-boundary N-grams `(2–5)` with sublinear TF-IDF and regularized linear classification (`LogisticRegression` / `LinearSVC` calibrated probabilities) to capture exact subword and transliterated spelling cues.
   * **Branch 2 — Cross-Lingual Transformer (`XLM-RoBERTa-base`):** Fine-tuned with class-balanced weighting to capture multilingual sentence-level semantics across code-mixed inputs.
   * **Branch 3 — Indic-Specialized Transformer (`IndicBERTv2-MLM-only`):** Fine-tuned specifically for Indian-language morphology, regional phrasing, and Romanized Indic script variations.
3. **Out-of-Fold Weighted Blending & Macro-F1 Post-Processing:** Combines out-of-fold probability distributions across all three branches and tunes per-class decision thresholds (including the `Class 55` OOS boundary) directly on validation Macro-F1.

---

## 🛠️ Tech Stack
- **NLP & Deep Learning:** PyTorch, Hugging Face `transformers` (`XLM-RoBERTa-base`, `IndicBERTv2-MLM-only`)
- **Classical ML & Feature Engineering:** Scikit-Learn (`TfidfVectorizer`, Word & Char-wb N-grams, Calibrated Linear Models)
- **Optimization & Evaluation:** Stratified K-Fold Cross-Validation, Out-of-Fold Probability Ensembling, Threshold Tuning for Macro-F1
- **Environment:** Python, Kaggle GPU Runtime

---

## ⚙️ Reproducing the Pipeline

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/krishnayyar999-sudo/](https://github.com/krishnayyar999-sudo/)<YOUR-REPO-NAME>.git
   cd <YOUR-REPO-NAME>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training & inference** (or inspect the full executed GPU run directly on **[Kaggle](https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07)**):
   ```bash
   python train.py
   python inference.py
   ```
   *(Update the script filenames above if your repository uses a `.ipynb` notebook or different entry script names.)*

---

## 👤 Author
**Krish Nayyar**  
B.Tech in Computer Science & Engineering — Guru Nanak Dev University, Amritsar  
- **GitHub:** [github.com/krishnayyar999-sudo](https://github.com/krishnayyar999-sudo)  
- **LinkedIn:** [linkedin.com/in/krish-nayyar-071240420](https://www.linkedin.com/in/krish-nayyar-071240420)
