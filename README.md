# Multilingual Intent Classification & Out-of-Scope Detection
### Anveshan Data Sprint 2.0 — NLP Track Submission

---

## 🔗 Quick Links
* **Live Kaggle Notebook (With Executed Outputs):** https://www.kaggle.com/code/krishnayyar27/notebookf20be6be07
* **Competition Platform:** Anveshan Data Sprint 2.0 (The National Institute of Engineering, Karnataka / Unstop & Kaggle)
* **Team Name:** `krishnayyar999`
* **Best Public Leaderboard Score:** **`0.77858`** (Macro-F1)

---

## 📌 Problem Statement
Conversational AI assistants operating across multilingual and code-mixed regions face two core challenges:
1. **Fine-Grained Intent Routing:** Accurately classifying noisy, short, transliterated, and code-mixed user queries across **55 distinct intent categories**.
2. **Out-of-Scope (OOS) Detection:** Reliably identifying unsupported or irrelevant queries (**Class 55**) so the system falls back gracefully instead of triggering false actions.

Because real-world intent distributions are heavily imbalanced, standard accuracy-driven models overfit to majority intents and underperform on rare tail classes. This solution directly optimizes **Macro-F1** across all **56 classes** (`0–55`).

---

## 🧠 Solution Architecture

Our pipeline combines **three diverse model families** (lexical linear models + cross-lingual transformer + Indic-specialized transformer) with metric-aligned probability calibration:

```text
                       ┌─────────────────────────────────────────────┐
                       │         Raw Multilingual User Query         │
                       └──────────────────────┬──────────────────────┘
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
                 ▼                            ▼                            ▼
┌─────────────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│   Branch 1: Statistical Lexical │ │ Branch 2: Cross-Lingual   │ │ Branch 3: Indic Semantic  │
│  Word (1–3) + Char_wb (2–5)     │ │    XLM-RoBERTa-base       │ │   IndicBERTv2-MLM-only    │
│           TF-
