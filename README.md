# Mango-recall_classification 🥭🧡

> Recall classification module for Mango.  
> Fine-tuned with 🤗 Huggingface Transformers.

---

## 📚 Overview

This module classifies Korean military conversations into two categories:  
It is used to detect **emotionally meaningful recalls** in Mango.

---

## 🏷️ Training Data

- **Total**: 200 examples
- **Label Definitions**:
  - **label 1** : 과거 회상, 감정적 반응 (울음, 무서움, 죄책감 등), 의미 있는 사건 (휴가, 복귀, 사고, 갈등 등)
  - **label 0** : 일상 보고, 정보 전달, 감정 없는 대화, 주변 묘사 등

---

## 🔗 Model Repository

You can access the trained model on Huggingface Hub here:  
[🤗 Huggingface Model Link](https://huggingface.co/kjsbrian/mango-recall-classifier/tree/main)

---

## 🚀 Quick Start

### Install Requirements

```bash
pip install transformers
pip install datasets
pip install huggingface_hub
