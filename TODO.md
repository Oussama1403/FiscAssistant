# todo.md

## Project: FiscAssistant Backend Transformation

This file lists all tasks needed to overhaul the FiscAssistant backend into a high-level, AI-driven chatbot for fiscal assistance in Tunisia, supporting French, English, Arabic, and Tunisian dialect. Tasks are grouped by category and prioritized for a logical development flow. Each task includes a description and estimated effort level (Low: <1 hour, Medium: 1–4 hours, High: >4 hours). The `nous-hermes2:10.7b-solar-q3_k_m` model is used, as it runs successfully on the target hardware (16GB RAM, RTX 4GB VRAM).

### 1. Environment Setup
- [x] **Task 1.1: Install Required Dependencies**
  - **Description**: Install Python libraries: `fastapi`, `uvicorn`, `llama-cpp-python`, `langchain`, `chromadb`, `sentence-transformers`, `pdfplumber`, `sympy`, `numpy`, `pandas`, `apscheduler`, `pydantic`, `pytest`, and `structlog`. Ensure compatibility with Python 3.10+.
  - **Effort**: Low
  - **Status**: Completed

- [x] **Task 1.2: Download Quantized Nous-Hermes-2 Model**
  - **Description**: Use the `nous-hermes2:10.7b-solar-q3_k_m` model, confirmed to work on the target hardware. Loaded with `llama.cpp` for hybrid CPU/GPU inference.
  - **Effort**: Medium
  - **Status**: Completed

### 2. Data Preparation
- [x] **Task 2.1: Collect Fiscal Data**
  - **Description**: Gather Tunisian fiscal data: tax rates (TVA, IRPP, CNSS), declaration deadlines (monthly, quarterly, annual), legal steps for enterprise creation (SUARL, SARL, etc.), penalty information, and sample fiscal forms (PDFs from DGI, CNSS). Store in a structured format (e.g., JSON or CSV).
  - **Effort**: High
  - **Dependencies**: None
  - **Notes**: Sources include DGI Tunisia website, CNSS, and official documents. Aim for ~100–200 entries.

- [x] **Task 2.2: Create Multilingual Dialogue Dataset**
  - **Description**: Compile a dataset of Q&A pairs in French, English, Arabic, and Tunisian dialect for fiscal tasks (e.g., “كيفاش نحسب تڤا؟” → “تڤا = المبلغ × النسبة ÷ 100”). Include ~200–300 examples covering greetings, calculations, and explanations.
  - **Effort**: High
  - **Dependencies**: None
  - **Notes**: Use Tunisian dialect terms (e.g., “تڤا”, “شنو”, “كيفاش”). Save as JSON for fine-tuning.

- [ ] **Task 2.3: Set Up ChromaDB Knowledge Base**
  - **Description**: Initialize a ChromaDB instance and populate it with fiscal data from Task 2.1 (tax rates, deadlines, etc.) using SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings. Test retrieval with sample queries (e.g., “TVA deadline”).
  - **Effort**: Medium
  - **Dependencies**: Task 2.1
  - **Notes**: Ensure data is indexed for multilingual queries.

### 3. LLM Optimization
- [ ] **Task 3.1: Integrate Nous-Hermes-2 with llama.cpp**
  - **Description**: Replace `Ollama` with `llama-cpp-python` in `main.py`. Load the `nous-hermes2:10.7b-solar-q3_k_m` model and configure for hybrid CPU/GPU inference (`n_gpu_layers=20`, `n_ctx=4096`). Test with sample inputs.
  - **Effort**: Medium
  - **Dependencies**: None
  - **Notes**: Monitor VRAM usage with `nvidia-smi` and adjust layers if needed.

- [ ] **Task 3.2: Fine-Tune Nous-Hermes-2 with LoRA**
  - **Description**: Fine-tune the model using LoRA on the dialogue dataset (Task 2.2) to improve fiscal knowledge and Tunisian dialect support. Use `peft` library and train on CPU or GPU (~1GB RAM needed). Save fine-tuned weights.
  - **Effort**: High
  - **Dependencies**: Tasks 2.2, 3.1
  - **Notes**: Use a small learning rate and ~100 epochs. Test fine-tuned model on dialect-specific inputs.

- [ ] **Task 3.3: Design Flexible Prompt Templates**
  - **Description**: Create LangChain prompt templates for different tasks (greeting, calculation, retrieval, explanation, quiz). Avoid rigid JSON outputs; use chain-of-thought prompting for reasoning. Test with diverse inputs.
  - **Effort**: Medium
  - **Dependencies**: Task 3.1
  - **Notes**: Example: “Reason through the user’s query step-by-step, then provide a concise response in {language}.”

### 4. Intent and Language Detection
- [ ] **Task 4.1: Implement Embedding-Based Language Detection**
  - **Description**: Replace `langdetect` and keyword-based detection with SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`). Generate embeddings for user inputs and compare to reference phrases in French, English, Arabic, and Tunisian dialect. Test accuracy on mixed-language inputs.
  - **Effort**: Medium
  - **Dependencies**: None
  - **Notes**: Use cosine similarity for language classification.

- [ ] **Task 4.2: Implement Intent Classification**
  - **Description**: Train a lightweight classifier (e.g., scikit-learn LogisticRegression) on SentenceTransformer embeddings to detect intents (greeting, VAT calculation, profit simulation, profile detection, etc.). Use ~50–100 labeled examples from Task 2.2. Integrate into `main.py`.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.2, 4.1
  - **Notes**: Intents should cover all requirements (A–I).

### 5. Retrieval-Augmented Generation (RAG)
- [ ] **Task 5.1: Integrate ChromaDB with LangChain**
  - **Description**: Create a LangChain retrieval chain to query ChromaDB based on user input embeddings. Return top-k relevant documents (e.g., tax rates, deadlines) and combine with LLM responses. Test with queries like “What’s the TVA rate for services?”
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 3.1, 4.1
  - **Notes**: Use LangChain’s `RetrievalQA` chain for simplicity.

- [ ] **Task 5.2: Add PDF Form Processing**
  - **Description**: Implement a PDFPlumber module to extract fields from fiscal forms (e.g., TVA declaration form). Store extracted data in ChromaDB and create a LangChain tool to guide users through form fields. Test with a sample DGI form.
  - **Effort**: High
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement B (form guidance).

### 6. Calculation Engine
- [ ] **Task 6.1: Implement Calculation Module**
  - **Description**: Create a calculation module using SymPy/NumPy/Pandas for VAT (`amount * rate / 100`), net profit (`(revenue - expenses) * (1 - tax_rate/100)`), profitability thresholds, and IRPP/CNSS contributions. Store tax tables in Pandas DataFrame. Test with sample inputs.
  - **Effort**: Medium
  - **Dependencies**: None
  - **Notes**: Use ChromaDB for tax rate lookups (Task 2.3).

- [ ] **Task 6.2: Integrate Calculations with LangChain**
  - **Description**: Create LangChain custom tools for calculations and integrate with the agent framework. Route calculation queries (e.g., “Calculate VAT for 100 TND at 7%”) to the module. Test accuracy and response formatting.
  - **Effort**: Medium
  - **Dependencies**: Tasks 3.1, 4.2, 6.1
  - **Notes**: Supports Requirements C and E.

### 7. Notifications and Scheduling
- [ ] **Task 7.1: Replace schedule with APScheduler**
  - **Description**: Replace `schedule` with APScheduler in `main.py`. Implement persistent scheduling for daily reminders (e.g., at 8:00 AM). Store reminders in SQLite database (extend `db.database`). Test with sample reminders.
  - **Effort**: Low
  - **Dependencies**: None
  - **Notes**: Supports Requirement G.

- [ ] **Task 7.2: Add Notification Delivery**
  - **Description**: Implement a WebSocket endpoint (`/ws/reminders`) to send real-time notifications to the React frontend. Optionally, add email notifications using `smtplib` (if internet available). Test delivery for deadlines.
  - **Effort**: Medium
  - **Dependencies**: Task 7.1
  - **Notes**: Enhances Requirement G.

### 8. Advanced Features
- [ ] **Task 8.1: Implement Quiz Mode**
  - **Description**: Create a LangChain chain to generate fiscal quiz questions from ChromaDB (e.g., “What’s the TVA rate for restaurants?”). Add a `/quiz` endpoint to handle quiz interactions. Test with 5–10 sample questions.
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement H.

- [ ] **Task 8.2: Add Enterprise Creation Guidance**
  - **Description**: Create a LangChain chain to provide step-by-step guidance for enterprise creation (e.g., SUARL registration steps) based on ChromaDB data. Test with queries like “How to start a SUARL?”
  - **Effort**: Medium
  - **Dependencies**: Tasks 2.3, 5.1
  - **Notes**: Supports Requirement F.

- [ ] **Task 8.3: Implement Fiscal Control Simulation**
  - **Description**: Create a LangChain chain to simulate fiscal control Q&A (e.g., “What documents are needed for a tax audit?”). Use ChromaDB for penalty and recourse data. Test with sample scenarios.
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
  - **Description**: Implement a `/upload_form` endpoint to handle PDF form uploads. Process files with PDFPlumber and return extracted fields or guidance. Test with a sample fiscal form.
  - **Effort**: Medium
  - **Dependencies**: Task 5.2
  - **Notes**: Supports Requirement B.

### 10. Testing and Optimization
- [ ] **Task 10.1: Write Unit Tests**
  - **Description**: Use Pytest to write unit tests for API endpoints (`/chat`, `/set_reminder`, `/ws/chat`), calculation module, and RAG responses. Test multilingual inputs and calculation accuracy.
  - **Effort**: High
  - **Dependencies**: Tasks 3.1, 5.1, 6.2, 7.1, 9.1
  - **Notes**: Aim for >80% code coverage.

- [ ] **Task 10.2: Optimize Performance**
  - **Description**: Monitor VRAM/RAM usage with `nvidia-smi` and `htop`. Adjust `llama.cpp` parameters (`n_gpu_layers`, batch size) and FastAPI workers to optimize for 4GB VRAM and 16GB RAM. Test response latency.
  - **Effort**: Medium
  - **Dependencies**: Task 3.1
  - **Notes**: Target <2s response time for most queries.

- [ ] **Task 10.3: Enhance Logging**
  - **Description**: Replace `logging` with `structlog` for structured logging. Add logs for intent detection, RAG retrieval, and errors. Implement a `/health` endpoint for monitoring. Test log output.
  - **Effort**: Low
  - **Dependencies**: None
  - **Notes**: Improves debugging and monitoring.

### 11. Deployment
- [ ] **Task 11.1: Set Up Docker Environment**
  - **Description**: Create a `Dockerfile` and `docker-compose.yml` to containerize FastAPI, `llama.cpp`, and dependencies. Use NVIDIA Docker for GPU support. Test locally to ensure compatibility with 16GB RAM and 4GB VRAM.
  - **Effort**: Medium
  - **Dependencies**: All prior tasks
  - **Notes**: Include a `.dockerignore` to exclude large files (e.g., model weights).

### 12. Documentation
- [ ] **Task 12.1: Document Backend Architecture**
  - **Description**: Create a system architecture diagram (e.g., using draw.io) showing FastAPI, LangChain, ChromaDB, LLM, and calculation engine. Write documentation for endpoints, setup, and usage.
  - **Effort**: Medium
  - **Dependencies**: All tasks
  - **Notes**: Include in graduation project report.

- [ ] **Task 12.2: Prepare Demo Script**
  - **Description**: Write a demo script showcasing key features: multilingual responses, calculations, form guidance, and reminders. Record a video or live demo with React frontend.
  - **Effort**: Medium
  - **Dependencies**: All tasks
  - **Notes**: Highlight Requirements A–I in the demo.

---

## Prioritization and Workflow
- **Start with**: Tasks 2.1 or 2.2 (data collection) to build the foundation for fiscal knowledge and dialect support.
- **Core AI**: Tasks 3.1, 3.2, 4.1, 4.2, 5.1 (LLM integration, fine-tuning, intent detection, RAG) for intelligence.
- **Features**: Tasks 5.2, 6.1, 6.2, 7.1, 7.2, 8.1–8.3 (calculations, notifications, advanced features) for requirements.
- **Polish**: Tasks 9.1, 9.2, 10.1–10.3 (frontend integration, testing) for professionalism.
- **Final Steps**: Tasks 11.1, 12.1, 12.2 (deployment, documentation) for project completion.

## Notes
- **Hardware**: Monitor 4GB VRAM and 16GB RAM usage during Tasks 3.1, 3.2, and 10.2.
- **Dataset**: Tasks 2.1 and 2.2 are critical for domain-specific accuracy and dialect support.
- **Testing**: Task 10.1 ensures reliability for your graduation project.
- **Documentation**: Tasks 12.1 and 12.2 are essential for a polished submission.

