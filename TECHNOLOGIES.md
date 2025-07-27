FiscAssistant: Technology Stack

This document outlines the final technologies used in the FiscAssistant chatbot, a tax-assistant application developed for a student’s graduation project (PFE). The stack is designed to deliver a robust, scalable, and user-friendly solution, meeting all project requirements (multilingual greetings, tax declarations, VAT calculations, penalties, profitability simulations, business creation, reminders, tax education, and multilingual support with Tunisian dialect approximation).

🧠 Core AI Model (LLM)

Nous-Hermes-2-Mixtral-8x7B-DPO
Description: A state-of-the-art large language model by Nous Research, fine-tuned for conversational tasks, reasoning, and multilingual capabilities.
Features:
Supports French, English, and Arabic natively.
Approximates Tunisian dialect using standard Arabic prompts (full dialect support requires fine-tuning, deferred for PFE).
32K token context window for robust multi-turn conversations.
4-bit quantized, requiring ~8–10GB RAM.


Purpose: Handles natural language understanding and generation for greetings (A), declarations (B), penalties (D), business creation (F), education (H), and multilingual interactions (I).
Deployment: Run locally via Ollama.



🧪 Backend

FastAPI
Description: A modern, high-performance Python web framework with asynchronous support.
Features:
Manages API routes for chat, calculations, and reminders.
Integrates with Ollama for LLM calls and LangChain for context management.
Provides automatic OpenAPI documentation for professional-grade APIs.


Purpose: Orchestrates all project requirements (A–I), including LLM interactions, custom logic, and user sessions.
Why FastAPI?: Faster and more scalable than Flask, with type safety and modern features, ideal for showcasing advanced skills.



🧠 LLM Toolkit

LangChain
Description: A Python framework for building applications with LLMs.
Features:
Manages conversation history using ConversationBufferMemory for context-aware responses.
Integrates tools for calculations (VAT, profitability) and reminders.
Supports task chaining (e.g., greet → detect profile → provide tax advice).


Purpose: Enhances requirements A (context-aware greetings), B, D, F, H (conversational tasks), I (multilingual context), and C, E, G (via tools).
Why LangChain?: Simplifies context management and tool integration, adding academic rigor to the PFE.



📐 Custom Fiscal Logic

Python with schedule and pydantic
Description: Pure Python code for reliable calculations and scheduling, enhanced with Pydantic for input validation.
Features:
Calculations: Implements VAT, IRPP, CNSS, profit margin, and rentability simulations using Python functions.
Pydantic: Validates user inputs (e.g., amounts, rates) for type safety in FastAPI endpoints.
Schedule: Manages in-memory reminders for tax deadlines, with console-based output (extendable to UI or email).


Purpose: Addresses requirements C (VAT calculations), E (profitability simulations), and G (reminders).
Example:from pydantic import BaseModel
def calculate_vat(amount: float, rate: float = 19) -> float:
    return amount * (rate / 100)





🌐 Frontend (UI)

React with Tailwind CSS
Description: A modern JavaScript library for building dynamic single-page applications, styled with Tailwind CSS.
Features:
Creates an interactive chat interface with language selection, conversation history, and reminder display.
Tailwind CSS provides utility-first styling for a professional look with minimal effort.
Deployed via CDN for simplicity (no complex build tools).


Purpose: Supports requirements A (user interaction), I (language selection), G (display reminders), and H (quiz interface).
Why React/Tailwind?: Offers a polished, scalable UI, elevating the PFE demo over basic HTML/CSS/JS.



🗄️ Database (Optional for Persistence)

SQLite
Description: A lightweight, serverless database integrated with Python’s sqlite3 module.
Features:
Stores conversation history and reminders for multi-session support.
Persists user profiles and language preferences.


Purpose: Enhances requirements A (profile persistence), G (reminder storage), and I (language preferences).
Why SQLite?: Simple setup, suitable for a student project, adds robustness without complexity.



🧳 Local Model Management

Ollama
Description: A tool for running large language models locally with GPU/CPU support.
Features:
Deploys Nous-Hermes-2-Mixtral-8x7B-DPO with minimal setup (ollama run).
No internet or external API dependencies, ensuring privacy and cost-free operation.


Purpose: Powers all LLM interactions for requirements A–I.
Why Ollama?: Simplifies local model execution, leveraging your PC’s capabilities.



📋 Summary of Requirements Coverage

A (Accueil & Orientation): Nous-Hermes-2 for multilingual greetings/profile detection; FastAPI/LangChain for context; SQLite for profile persistence.
B (Déclarations Fiscales): LLM for explanations; Python for date checks.
C (TVA & Régimes Fiscaux): LLM for rate/regime advice; Python/Pydantic for calculations.
D (Infractions & Risques): LLM for penalties/audit Q&A.
E (Simulation de Rentabilité): LLM for concepts; Python for calculations.
F (Création d’Entreprise): LLM for steps/obligations.
G (Rappels et Notifications): LLM for confirmation; schedule/SQLite for reminders.
H (Formation & Éducation Fiscale): LLM for explanations/quizzes; Python for links.
I (Multilingue & Localisé): LLM for French, English, Arabic; dialect approximation.

🚀 Why This Stack?

Robustness: Nous-Hermes-2 outperforms prior options (bloom-1b7, Gemini) in conversational and multilingual tasks.
Scalability: FastAPI, LangChain, and React ensure a modern, extensible architecture.
Polish: React/Tailwind and SQLite add professional-grade UI and persistence.
Privacy: Local execution via Ollama avoids external API risks.
Academic Impact: Advanced technologies (FastAPI, LangChain) showcase your skills, impressing PFE evaluators.

🛠️ Setup Instructions

Ollama: Install via curl https://ollama.ai/install.sh | sh (Linux) or download for macOS/Windows. Pull model: ollama pull nous-hermes2:8x7b-dpo-q4_0.
Dependencies: pip install fastapi uvicorn langchain-community langchain-core schedule pydantic sqlite3.
Frontend: Use React/Tailwind via CDN:<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="https://cdn.tailwindcss.com"></script>


Database: Initialize SQLite with Python’s sqlite3 for conversation/reminder storage.
Run: Start FastAPI with uvicorn main:app --reload.

📝 PFE Documentation Notes

Highlight: Modern stack (FastAPI, LangChain, React), local LLM execution, and robust calculations.
Limitations: Tunisian dialect approximation, console-based reminders (extendable to email/UI).
Future Work: Fine-tune dialect, add external notifications (e.g., email via smtplib), deploy to cloud.
