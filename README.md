# ⚡ Proposera AI  
## Autonomous RFP Engineering System  
### Submission for TurboHaQ26 GenAI Hackathon 2026  

Transforming weeks of manual B2B sales cycles into seconds of autonomous precision.

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

---

## 📽️ Project Overview

**Proposera AI** is a multi-agent autonomous RFP engine that automates the most painful bottleneck in enterprise B2B sales: responding to complex Requests for Proposals (RFPs).

Built on a modular **Neural Midnight Architecture**, it combines:

- Retrieval-Augmented Generation (RAG)
- Agentic Web Research
- Local Vector Search (FAISS)
- Autonomous Document Synthesis
- Secure SMTP Dispatch

to generate compliant, enterprise-ready proposals in minutes — not weeks.

---
<img width="1919" height="1003" alt="image" src="https://github.com/user-attachments/assets/29f0fc13-e4ad-4242-a795-9668c740ec79" />


## 🔴 The Problem

Enterprise sales engineers spend **40+ hours per RFP**:

- Manually cross-referencing security documents
- Reviewing pricing sheets
- Matching past case studies
- Ensuring compliance accuracy

This slows revenue velocity, increases human error, and creates massive operational overhead.

---

## 🟢 The Solution

Proposera AI acts as a **Virtual Sales Engineer**.

It autonomously:

✔ Ingests internal company knowledge (Security, Pricing, Case Studies)  
✔ Identifies client names & requirements from raw RFP uploads  
✔ Performs live web research for competitive validation  
✔ Synthesizes complete, formatted `.docx` enterprise proposals  
✔ Dispatches proposals directly via secure SMTP  

---

# 🚀 Multi-RFP Batch Engine (Key Differentiator)

Unlike traditional single-document systems, Proposera AI supports:

## 🔁 Processing up to **10 RFPs simultaneously**

- Upload up to **10 PDF RFP files in one batch**
- Parallel ingestion & requirement extraction
- Sequential proposal generation
- Independent email dispatch per client
- Automatic failure isolation (one RFP failing does not stop others)

This transforms sales teams into true high-velocity proposal engines.

---

# 🔥 Core Features

### 🧠 Enterprise RAG Brain
Powered by FAISS + HuggingFace embeddings for ultra-fast local semantic retrieval.

### 🌐 Autonomous Web Spy
Agentic DuckDuckGo integration for real-time market intelligence.

### 📦 Batch Processing Engine
Handles up to **10 PDFs in a single run**, enabling bulk RFP execution.

### 🛡️ Neural Scrubbing Layer
Custom Unicode-safe SMTP dispatch to prevent ligature (“ﬃ”) email failures.

### 📬 End-to-End Dispatch
SMTP bridge integration for automated email delivery with attachments.

### 🎨 Neural Midnight UI
Custom neon-themed Streamlit dashboard with grid overlays and glow effects.

---

# 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.12 |
| LLM | Google Gemini 1.5 Flash / Gemini 2.0 |
| Orchestration | LangChain (LCEL) |
| Vector Database | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| UI | Streamlit (Custom CSS Injected) |
| Search Agent | DuckDuckGo |
| Document Generation | python-docx |
| Email Dispatch | SMTP (Gmail App Password) |

---

# 🧩 System Architecture

1. Knowledge Base → Chunking → FAISS Index  
2. RFP Upload → Identity Extraction  
3. Agentic Web Research  
4. Context Retrieval via RAG  
5. Proposal Drafting via LLM  
6. `.docx` Generation  
7. Secure SMTP Dispatch  

---

# 🚀 Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/proposera-ai.git
cd proposera-ai
