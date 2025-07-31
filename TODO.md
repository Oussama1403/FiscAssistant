# TODO.md

## Project: FiscAssistant Backend Transformation

This file lists all tasks needed to overhaul the FiscAssistant backend into a high-level, AI-driven chatbot for fiscal assistance in Tunisia, supporting French, English, Arabic, and Tunisian dialect. Tasks are grouped by category and prioritized for a logical development flow. Each task includes a description and estimated effort level (Low: <1 hour, Medium: 1–4 hours, High: >4 hours). The `nous-hermes2:10.7b-solar-q3_k_m` model is used, as it runs successfully on the target hardware (16GB RAM, RTX 4GB VRAM). Fine-tuning is not accessible, so the focus is on enhancing RAG and other components.

### 1. Environment Setup
- [x] **Task 1.1: Install Required Dependencies**
  - **Description**: Install Python libraries: `fastapi`, `uvicorn`, `llama-cpp-python`, `langchain`, `chromadb`, `sentence-transformers`, `pdfplumber`, `sympy`, `numpy`, `pandas`, `apscheduler`, `pydantic`, `pytest`, `structlog`, and `scikit-learn`. Ensure compatibility with Python 3.10+.
  - **Effort**: Low
  - **Status**: Completed

- [x] **Task 1.2: Download Quantized Nous-Hermes-2 Model**
  - **Description**: Use the `nous-hermes2:10.7b-solar-q3_k_m` model, confirmed to work on the target hardware. Loaded with `llama.cpp` for hybrid CPU/GPU inference.
  - **Effort**: Medium
  - **Status**: Completed

### 2. Data Preparation
- [x] **Task 2.1: Collect Fiscal Data**
  - **Description**: Gather Tunisian fiscal data: tax rates (TVA, IRPP, CNSS), declaration deadlines, legal steps for enterprise creation (SUARL, SARL), penalties, and sample fiscal forms (PDFs from DGI, CNSS). Stored in `fiscal_data.json` with 314 entries in JSON format.
  - **Effort**: High
  - **Dependencies**: None
  - **Notes**: Sources include DGI Tunisia, CNSS, and official documents. Data includes multilingual descriptions (en, fr, ar, tn).

- [x] **Task 2.2: Create Multilingual Dialogue Dataset**
  - **Description**: Compile a dataset of 285 Q&A pairs in French, English, Arabic, and Tunisian dialect for fiscal tasks (e.g., “كيفاش نحسب تڤا؟” → “تڤا = المبلغ × النسبة ÷ 100”). Includes intents like `query_penalty`, `query_deadline`, `greeting`. Saved as `dialogue_dataset.json`.
  - **Effort**: High
  - **Dependencies**: None
  - **Notes**: Emphasizes Tunisian dialect terms (e.g., “تڤا”, “شنو”). Used for language and intent classification.

- [x] **Task 2.3: Set Up ChromaDB Knowledge Base**
  - **Description**: Initialize a ChromaDB instance and populate with fiscal data from `fiscal_data.json` and Q&A pairs from `dialogue_dataset.json` using SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings. Test retrieval with sample queries (e.g., “TVA deadline”).
  - **Effort**: Medium
  - **Dependencies**: Task 2.1, 2.2
  - **Notes**: Indexed for multilingual queries with metadata (language, intent, category).

### 3. LLM Optimization
- [x] **Task 3.1: Integrate Nous-Hermes-2 with Ollama**
  - **Description**: Integrate the `nous-hermes2:10.7b-solar-q3_k_m` model with Ollama for inference. Configured with `n_gpu_layers=10` to fit 4GB VRAM.
  - **Effort**: Medium
  - **Dependencies**: Task 1.2
  - **Status**: Completed

- [ ] **Task 3.2: Fine-Tune Nous-Hermes-2 with LoRA** *(Removed)*
  - **Description**: Fine-tuning is not accessible due to hardware or resource constraints. Replaced with enhanced RAG (Task 5.1) to improve fiscal knowledge and dialect support using `dialogue_dataset.json`.
  - **Effort**: N/A
  - **Dependencies**: N/A
  - **Notes**: Focus on RAG and intent classification instead.

- [x] **Task 3.3: Design Flexible Prompt Templates**
  - **Description**: Create LangChain prompt templates for intents (greeting, calculation, retrieval, quiz, etc.). Use chain-of-thought prompting for reasoning, avoiding rigid JSON outputs. Tested with diverse inputs in `main2.py`.
  - **Effort**: Medium
  - **Dependencies**: Task 3.1
  - **Notes**: Example: “Identify intent, use context, respond in {language}.” Supports Tunisian dialect for `tn`.

### 4. Intent and Language Detection
- [x] **Task 4.1: Implement Embedding-Based Language Detection**
  - **Description**: Replace cosine similarity with a LogisticRegression classifier trained on `dialogue_dataset.json` (285 entries) using SentenceTransformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`). Preprocess inputs (lowercase, standardize dialect terms like “كيفاش” → “كيف”). Fallback to conversation history if confidence < 0.5. Test on mixed-language inputs (e.g., “Bonjour, كيفاش نحسب تڤا؟”).
  - **Effort**: Medium
  - **Dependencies**: Task 2.2
  - **Notes**: Improves accuracy for Tunisian dialect and mixed inputs. Avoid LLM fallback to reduce latency.

- [x] **Task 4.2: Implement Intent Classification**
  - **Description**: Train a DistilBertForSequenceClassification classifier on `dialogue_dataset.json` using SentenceTransformer embeddings to detect intents (`query_penalty`, `query_deadline`, `greeting`, etc.). Integrate into `main2.py` to route queries to RAG or calculation tools. Test with ~50 examples per intent.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.2, 4.1
  - **Notes**: Supports all requirements (A–I) by mapping intents to actions.

### 5. Retrieval-Augmented Generation (RAG)
- [x] **Task 5.1: Integrate ChromaDB with LangChain**
  - **Description**: Create a LangChain `RetrievalQA` chain to query ChromaDB using `fiscal_data.json` (314 entries) and `dialogue_dataset.json` (285 entries). Filter by language and intent (e.g., `query_deadline` → `category: deadline`). Return top-3 documents and combine with LLM responses. Test with queries like “What’s the TVA rate for services?”.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 3.1, 4.1, 4.2
  - **Notes**: Enhances response accuracy without fine-tuning.

- [ ] **Task 5.2: Add PDF Form Processing**
  - **Description**: Implement a `pdfplumber` module to extract fields from fiscal forms (e.g., TVA declaration). Store in ChromaDB and create a LangChain tool to guide users through fields. Add `/upload_form` endpoint. Test with a sample DGI form.
  - **Effort**: High
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement B (form guidance).

### 6. Calculation Engine
- [ ] **Task 6.1: Implement Calculation Module**
  - **Description**: Create a module using Pandas/SymPy/NumPy for VAT (`amount * rate / 100`), net profit (`(revenue - expenses) * (1 - tax_rate/100)`), IRPP, CNSS, and profitability thresholds. Use `fiscal_data.json` for tax rates (e.g., 10% CIT for craft). Test with sample inputs.
  - **Effort**: Medium
  - **Dependencies**: Task 2.3
  - **Notes**: Supports Requirements C and E.

- [ ] **Task 6.2: Integrate Calculations with LangChain**
  - **Description**: Create LangChain tools for calculations (VAT, profit, IRPP, CNSS) and integrate with the agent framework. Route `calculate_tax` intent queries to the module. Test accuracy and response formatting (e.g., “Calculate VAT for 100 TND at 7%”).
  - **Effort**: Medium
  - **Dependencies**: Tasks 3.1, 4.2, 6.1
  - **Notes**: Supports Requirements C and E.

### 7. Notifications and Scheduling
- [ ] **Task 7.1: Replace schedule with APScheduler**
  - **Description**: Replace `schedule` with APScheduler in `main2.py`. Implement persistent scheduling for daily reminders (e.g., at 8:00 AM). Store reminders in SQLite (extend `db.database`). Test with sample reminders for deadlines like “Amnistie Sociale” (2025-12-31).
  - **Effort**: Low
  - **Dependencies**: None
  - **Notes**: Supports Requirement G.

- [ ] **Task 7.2: Add Notification Delivery**
  - **Description**: Implement a `/ws/reminders` WebSocket endpoint to send real-time notifications to the React frontend. Optionally, add email notifications using `smtplib` (if internet available). Test delivery for deadlines from `fiscal_data.json`.
  - **Effort**: Medium
  - **Dependencies**: Task 7.1
  - **Notes**: Enhances Requirement G.

### 8. Advanced Features
- [ ] **Task 8.1: Implement Quiz Mode**
  - **Description**: Create a LangChain chain to generate fiscal quiz questions from `fiscal_data.json` and `dialogue_dataset.json` (e.g., “What’s the TVA rate for restaurants?”). Add a `/quiz` endpoint. Test with 5–10 questions.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement H.

- [ ] **Task 8.2: Add Enterprise Creation Guidance**
  - **Description**: Create a LangChain chain to provide step-by-step guidance for enterprise creation (e.g., SUARL steps) using `fiscal_data.json` (`category: legal_step`). Test with queries like “How to start a SUARL?”.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement F.

- [ ] **Task 8.3: Implement Fiscal Control Simulation**
  - **Description**: Create a LangChain chain to simulate fiscal control Q&A (e.g., “What documents for a tax audit?”) using `fiscal_data.json` (`category: penalty`). Test with sample scenarios.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement D.

### 9. Frontend Integration
- [ ] **Task 9.1: Add WebSocket Endpoint**
  - **Description**: Implement a `/ws/chat` WebSocket endpoint in FastAPI for real-time chat with the React frontend. Stream LLM responses and handle conversation history. Test with React integration.
  - **Effort**: Medium
  - **Dependencies**: Task 3.1
  - **Notes**: Improves user experience.

- [ ] **Task 9.2: Add File Upload Endpoint**
  - **Description**: Implement a `/upload_form` endpoint to handle PDF form uploads. Process with `pdfplumber` and return extracted fields or guidance. Test with a sample fiscal form.
  - **Effort**: Medium
  - **Dependencies**: Task 5.2
  - **Notes**: Supports Requirement B.

### 10. Testing and Optimization
- [ ] **Task 10.1: Write Unit Tests**
  - **Description**: Use Pytest to write unit tests for endpoints (`/chat`, `/set_reminder`, `/ws/chat`, `/quiz`, `/upload_form`), calculation module, and RAG responses. Test multilingual inputs (e.g., “كيفاش نحسب تڤا؟”) and intents (`query_penalty`, `calculate_tax`). Aim for >80% coverage.
  - **Effort**: High
  - **Dependencies**: Tasks 3.1, 5.1, 6.2, 7.1, 9.1
  - **Notes**: Ensures reliability for graduation project.

- [ ] **Task 10.2: Optimize Performance**
  - **Description**: Monitor VRAM/RAM with `nvidia-smi` and `htop`. Adjust `llama.cpp` parameters (`n_gpu_layers=10`, batch_size=2) and FastAPI workers (limit to 1) for 4GB VRAM and 16GB RAM. Test response latency (<2s).
  - **Effort**: Medium
  - **Dependencies**: Task 3.1
  - **Notes**: Critical for constrained hardware.

- [ ] **Task 10.3: Enhance Logging**
  - **Description**: Replace `logging` with `structlog` for structured JSON logging. Log intent detection, RAG retrieval, and errors. Implement a `/health` endpoint for monitoring. Test log output.
  - **Effort**: Low
  - **Dependencies**: None
  - **Notes**: Improves debugging.

### 11. Deployment
- [ ] **Task 11.1: Set Up Docker Environment**
  - **Description**: Create a `Dockerfile` and `docker-compose.yml` to containerize FastAPI, `llama.cpp`, and dependencies. Use NVIDIA Docker for GPU support. Test locally for 16GB RAM and 4GB VRAM.
  - **Effort**: Medium
  - **Dependencies**: All prior tasks
  - **Notes**: Include `.dockerignore` to exclude large files (e.g., model weights).

### 12. Documentation
- [ ] **Task 12.1: Document Backend Architecture**
  - **Description**: Create a system architecture diagram (e.g., using draw.io) showing FastAPI, LangChain, ChromaDB, LLM, and calculation engine. Document endpoints (`/chat`, `/quiz`, `/upload_form`) and setup.
  - **Effort**: Medium
  - **Dependencies**: All tasks
  - **Notes**: Include in graduation project report.

- [ ] **Task 12.2: Prepare Demo Script**
  - **Description**: Write a demo script showcasing multilingual responses, calculations (VAT, profit), form guidance, and reminders. Record a video or live demo with React frontend.
  - **Effort**: Medium
  - **Dependencies**: All tasks
  - **Notes**: Highlight Requirements A–I.

