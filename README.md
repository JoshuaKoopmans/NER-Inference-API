# 🧠 NER-Serve — Named Entity Recognition API

A lightweight, production-ready **NER (Named Entity Recognition)** API built with **FastAPI**, **GLiNER**, and **Hugging Face** — reproducible using **uv**.

---

## 🚀 Features
- 🔥 Serve state-of-the-art NER models via FastAPI  
- 🌍 Multilingual, cross-domain (GLiNER v2.1)  
- 🔁 Easily switch or update models  
- 💾 Offline cache via Hugging Face  
- 🧩 One-step run with `uv` 

---

## ⚙️ Setup (Local)

### 1. Clone and enter

### 2. Create and activate environment
```bash
uv sync
source .venv/bin/activate
```

### 3. Run API
```bash
uvicorn main:app --reload
```

Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Configuration
Edit `config.yaml` to select your model and entities:
```yaml
model:
  name: gliner_multi-v2.1
  labels: ["person", "organization", "location", "weapon", "country"]
```

---

## 🔍 Example

**Request**
```bash
curl -X POST "http://127.0.0.1:8000/ner"   -H "Content-Type: application/json"   -d '{"text": "Apple is looking at buying U.K. startup for $1 billion"}'
```

**Response**
```json
[
  {"entity": "Apple", "label": "organization", "score": 0.98},
  {"entity": "U.K.", "label": "country", "score": 0.95}
]
```


## 🧾 License
MIT License — free for personal and commercial use.

---

> “Fast, multilingual, and ready for production.”
