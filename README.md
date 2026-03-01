# 2026 AI Engineering Toolkit🔥

**Build better LLM apps — faster, smarter, production-ready.**

A curated, list of 100+ libraries and frameworks for AI engineers building with Large Language Models. This toolkit includes battle-tested tools, frameworks, templates, and reference implementations for developing, deploying, and optimizing LLM-powered systems.

[![Toolkit banner](https://github.com/codedspaces/demo-2/blob/d9442b179eba2856e8c6e62bb1c6a1bb8c676b89/2.jpg?raw=true)](https://aiengineering.beehiiv.com/subscribe)

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

## 🧭 Navigator

> **Curated by AI Engineers for AI Engineers.** 🚀
> 100+ production-ready tools for the 2026 LLM lifecycle.

---

### 🛠️ Infrastructure & Core Tooling
*   [**Vector Databases**](#vector-databases) — High-performance retrieval & storage.
*   [**Orchestration & Workflows**](#orchestration--workflows) — Chain together complex logic.
*   [**PDF & Data Extraction**](#pdf-extraction-tools) — Structure the unstructured.
*   [**RAG Frameworks**](#rag-retrieval-augmented-generation) — Enhance LLMs with external knowledge.

### 🔬 Experimentation & Monitoring
*   [**Evaluation & Testing**](#evaluation--testing) — Benchmarking, unit testing, and quality assurance.
*   [**Model Management**](#model-management) — Versioning, tracking, and lifecycle ops.
*   [**Observability & Data Collection**](#data-collection--web-scraping) — Scraping, crawlers, and data pipelines.

### 🧠 Advanced Agentic & Logic
*   [**Agent Frameworks**](#-agent-frameworks) — Multi-agent systems and autonomous workflows.
*   [**Memory Management**](#memory-management) — Persistence, context, and semantic recall.
*   [**Synthetic Data Generation**](#synthetic-data-generation) — High-quality data pipelines.

### ⚡ Optimization & Deployment
*   [**LLM Training & Tuning**](#llm-training-and-fine-tuning) — Parameter-efficient fine-tuning and merging.
*   [**Inference & Serving**](#open-source-llm-inference) — Local hosting, quantization, and platforms.
*   [**Safety & Security**](#llm-safety--security) — Guardrails, red teaming, and scanning.
*   [**Structured Generation**](#structured-output--constraints) — Guided outputs and constraints.
*   [**Prompt Engineering**](#prompt-optimization--compression) — Optimization and compression strategies.

### 📚 Community & Learning
*   [**Contributing**](#-contributing) — Join the community.

## 🛠️ Tooling for AI Engineers

### Vector Databases

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Pinecone](https://www.pinecone.io/) | Managed vector database for production AI applications | API/SDK | Commercial |
| [Weaviate](https://github.com/weaviate/weaviate) | Open-source vector database with GraphQL API | Go | BSD-3 | 
| [Qdrant](https://github.com/qdrant/qdrant) | Vector similarity search engine with extended filtering | Rust | Apache-2.0 |
| [Chroma](https://github.com/chroma-core/chroma) | Open-source embedding database for LLM apps | Python | Apache-2.0 |
| [Milvus](https://github.com/milvus-io/milvus) | Cloud-native vector database for scalable similarity search | Go/C++ | Apache-2.0 | 
| [FAISS](https://github.com/facebookresearch/faiss) | Library for efficient similarity search and clustering | C++/Python | MIT | 
| [Deep Lake](https://github.com/activeloopai/deeplake) | AI-native data lake with versioned datasets, optimized for embeddings and multimodal storage | Python | Apache-2.0 | 
| [Vectara](https://github.com/vectara) | Managed RAG platform with APIs for retrieval and generation | Python/Go | Commercial |
| [SQLite-Vec](https://github.com/asg017/sqlite-vec) | A vector search SQLite extension that runs anywhere | C/JS/Python | MIT/Apache |

### Orchestration & Workflows

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | Framework for developing LLM applications | Python/JS | MIT | 
| [LlamaIndex](https://github.com/run-llama/llama_index) | Data framework for LLM applications | Python | MIT | 
| [Haystack](https://github.com/deepset-ai/haystack) | End-to-end NLP framework for production | Python | Apache-2.0 | 
| [DSPy](https://github.com/stanfordnlp/dspy) | Framework for algorithmically optimizing LM prompts | Python | MIT |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | SDK for integrating AI into conventional programming languages | C#/Python/Java | MIT | 
| [Langflow](https://github.com/langflow-ai/langflow) | Visual no-code platform for building and deploying LLM workflows | Python/TypeScript | MIT |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag-and-drop UI for creating LLM chains and agents | TypeScript | MIT |
| [Promptflow](https://github.com/microsoft/promptflow) | Workflow orchestration for LLM pipelines, evaluation, and deployment | Python | MIT |
| [Llama Stack](https://github.com/meta-llama/llama-stack) | Meta's official framework for building applications specifically with Llama models | Python/Many | MIT |
| [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) | Accelerates unstructured data preparation for LLM application developers | Python | Apache-2.0 |
| [Griptape](https://github.com/griptape-ai/griptape) | Modular Python framework for building complex AI-powered applications | Python | Apache-2.0 |

### PDF Extraction Tools

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Docling](https://github.com/docling-project/docling) | AI-powered toolkit converting PDF, DOCX, PPTX, HTML, images into structured JSON/Markdown with layout, OCR, table, and code recognition | Python | MIT |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Drill through PDFs at a character level, extract text & tables, and visually debug extraction | Python | MIT | 
| [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) | Lightweight, high-performance PDF parser for text/image extraction and manipulation | Python / C | AGPL-3.0 |
| [PDF.js](https://github.com/mozilla/pdf.js) | Browser-based PDF renderer with text extraction capabilities | JavaScript | Apache-2.0 | 
| [Camelot](https://github.com/camelot-dev/camelot) | Extracts structured tabular data from PDFs into DataFrames and CSVs | Python | MIT |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | Parse PDFs, DOCX, HTML into structured JSON for LLM workflows | Python | Apache-2.0 |
| [pdfminer.six](https://github.com/pdfminer/pdfminer.six) | Detailed PDF text extraction and layout analysis | Python | MIT |
| [Llama Parse](https://github.com/run-llama/llama_parse) | Structured parsing of PDFs and documents optimized for LLMs | Python | Apache-2.0 |
| [MegaParse](https://github.com/megaparse/megaparse) | Universal parser for PDFs, HTML, and semi-structured documents | Python | Apache-2.0 |
| [ExtractThinker](https://github.com/extract-thinker/extract-thinker) | Intelligent document extraction framework with schema mapping | Python | MIT |
| [PyMuPDF4LLM](https://github.com/JKamlah/pyMuPDF4LLM) | Wrapper around PyMuPDF for LLM-ready text, tables, and image extraction | Python | Apache-2.0 |

### RAG (Retrieval-Augmented Generation)

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [RAGFlow](https://github.com/infiniflow/ragflow) | Open-source RAG engine based on deep document understanding | Python | Apache-2.0 | 
| [Verba](https://github.com/weaviate/Verba) | Retrieval Augmented Generation (RAG) chatbot | Python | BSD-3 | 
| [PrivateGPT](https://github.com/imartinez/privateGPT) | Interact with documents using local LLMs | Python | Apache-2.0 | 
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | All-in-one AI application for any LLM | JavaScript | MIT |
| [Quivr](https://github.com/QuivrHQ/quivr) | Your GenAI second brain | Python/TypeScript | Apache-2.0 |
| [Jina](https://github.com/jina-ai/jina) | Cloud-native neural search framework for multimodal RAG | Python | Apache-2.0 |
| [txtai](https://github.com/neuml/txtai) | All-in-one embeddings database for semantic search and workflows | Python | Apache-2.0 |
| [FastGraph RAG](https://github.com/circlemind-ai/fast-graphrag) | Graph-based RAG framework for structured retrieval | Python | MIT |
| [Chonkie](https://github.com/chonkie-inc/chonkie) | Chunking utility for efficient document processing in RAG | Python | MIT |
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Low-latency RAG research toolkit with modular design and benchmarks | Python | MIT |
| [Llmware](https://github.com/llmware-ai/llmware) | Lightweight framework for building RAG-based apps | Python | Apache-2.0 |
| [RAG to Riches](https://github.com/SciPhi-AI/R2R) | Framework to build, scale, and deploy state-of-the-art RAG applications | Python | MIT |
| [BeyondLLM](https://github.com/aiplanethub/beyondllm) | All-in-one toolkit for experimentation, evaluation, and deployment of RAG | Python | Apache-2.0 |
| [fastRAG](https://github.com/IntelLabs/fastRAG) | Research framework for efficient and optimized RAG pipelines | Python | Apache-2.0 |
| [Rerankers](https://github.com/AnswerDotAI/rerankers) | Lightweight unified API for various text reranking models | Python | Apache-2.0 |

### Evaluation & Testing

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Evals](https://github.com/openai/evals) | OpenAI's framework for creating and running LLM evaluations | Python | MIT |
| [Ragas](https://github.com/explodinggradients/ragas) | Evaluation framework for RAG pipelines | Python | Apache-2.0 |
| [Opik](https://github.com/comet-ml/opik) | DevOps platform for evaluation, monitoring, and observability | Python | Apache-2.0 |
| [Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLM, vision, language, and tabular models | Python | Apache-2.0 |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation framework for unit testing LLM outputs | Python | Apache-2.0 |
| [TruLens](https://github.com/truera/trulens) | Evaluation and tracking for LLM experiments | Python | MIT |
| [UpTrain](https://github.com/uptrain-ai/uptrain) | Open-source tool to evaluate and improve LLM applications | Python | Apache-2.0 |
| [Giskard](https://github.com/Giskard-AI/giskard) | Testing framework for ML/LLMs with bias and robustness checks | Python | Apache-2.0 |
| [Weave](https://github.com/wandb/weave) | Experiment tracking, debugging, and logging for LLM workflows | Python | Apache-2.0 |
| [Lighteval](https://github.com/huggingface/lighteval) | Lightweight and fast evaluation framework from Hugging Face | Python | Apache-2.0 |
| [AgentEvals](https://github.com/langchain-ai/agentevals) | Utilities for evaluating the performance of LLM agents | Python | MIT |
| [PydanticAI Evals](https://ai.pydantic.dev/evals/) | Evaluation framework designed for systematic performance tracking | Python | MIT |
| [UQLM](https://github.com/cvs-health/uqlm) | Package for zero-resource LLM hallucination quantification | Python | Apache-2.0 |
| [PromptBench](https://github.com/microsoft/promptbench) | Unified evaluation framework for large language models | Python | Apache-2.0 |
| [LangTest](https://github.com/JohnSnowLabs/langtest) | Tool for testing model accuracy, bias, and robustness | Python | Apache-2.0 |
| [EvalPlus](https://github.com/evalplus/evalplus) | Rigorous evaluation framework specifically for LLM code generation | Python | Apache-2.0 |
| [judges](https://github.com/quotient-ai/judges) | Small library of LLM-based judges for automated evaluation | Python | Apache-2.0 |
| [AgentOps](https://github.com/AgentOps-AI/agentops) | SDK for monitoring and observability of AI agents in production | Python | MIT |

### Model Management

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Client library for Hugging Face Hub | Python | Apache-2.0 | 
| [MLflow](https://github.com/mlflow/mlflow) | Platform for ML lifecycle management | Python | Apache-2.0 |
| [Weights & Biases](https://github.com/wandb/wandb) | Developer tools for ML | Python | MIT |
| [DVC](https://github.com/iterative/dvc) | Data version control for ML projects | Python | Apache-2.0 |
| [ClearML](https://github.com/allegroai/clearml) | End-to-end MLOps platform with LLM support | Python | Apache-2.0 |

### Data Collection & Web Scraping

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Firecrawl](https://github.com/mendableai/firecrawl) | AI-powered web crawler that extracts and structures content for LLM pipelines | TypeScript | MIT |
| [Scrapy](https://github.com/scrapy/scrapy) | Fast, high-level web crawling & scraping framework | Python | BSD-3 |
| [Playwright](https://github.com/microsoft/playwright) | Web automation & scraping with headless browsers | TypeScript/Python/Java/.NET | Apache-2.0 | 
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) | Easy HTML/XML parsing for quick scraping tasks | Python | MIT |
| [Selenium](https://github.com/SeleniumHQ/selenium) | Browser automation framework (supports scraping) | Multiple | Apache-2.0 |
| [Newspaper3k](https://github.com/codelucas/newspaper) | News & article extraction library | Python | MIT |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) | Fast, lightweight, and modern web crawling & scraping library for AI data pipelines | Python | Apache-2.0 |
| [Colly](https://github.com/gocolly/colly) | High-performance scraping framework for Go | Go | BSD-2 |
| [Trafilatura](https://github.com/adbar/trafilatura) | Extract clean text from web pages for LLM training corpora | Python | MIT |
| [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai) | Use LLMs to extract structured data from websites and documents | Python | MIT |
| [Crawlee](https://github.com/apify/crawlee) | Web scraping and crawling framework for large-scale data collection | TypeScript | Apache-2.0 |

### Memory Management

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [mem0](https://github.com/mem0ai/mem0) | The smart memory layer for AI applications that improves over time | Python | Apache-2.0 |
| [Memoripy](https://github.com/caspianmoon/memoripy) | AI memory layer with short- and long-term storage and semantic clustering | Python | MIT |
| [Memobase](https://github.com/memodb-io/memobase) | User profile-based memory system for personalized Generative AI applications | Python | Apache-2.0 |
| [Memary](https://github.com/kingjulio8238/Memary) | Open-source memory layer designed for autonomous agents | Python | MIT |

### Synthetic Data Generation

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [DataDreamer](https://github.com/datadreamer-dev/DataDreamer) | Library for prompting, synthetic data generation, and training workflows | Python | MIT |
| [fabricator](https://github.com/flairNLP/fabricator) | Flexible open-source framework to generate datasets with LLMs | Python | Apache-2.0 |
| [Promptwright](https://github.com/stacklok/promptwright) | Specialized library for synthetic dataset generation | Python | Apache-2.0 |
| [EasyInstruct](https://github.com/zjunlp/EasyInstruct) | Easy-to-use instruction processing framework for LLMs | Python | Apache-2.0 |

## 🤖 Agent Frameworks

| Framework | Description | Language | License |
|-----------|-------------|----------|---------|
| [Google's ADK](https://google.github.io/adk-docs/) | Flexible and modular framework for developing and deploying AI agents | Python / Java | Apache-2.0 |
| [AutoGen](https://github.com/microsoft/autogen) | Multi-agent conversation framework | Python | CC-BY-4.0 | 
| [CrewAI](https://github.com/joaomdmoura/crewAI) | Framework for orchestrating role-playing autonomous AI agents | Python | MIT | 
| [LangGraph](https://github.com/langchain-ai/langgraph) | Build resilient language agents as graphs | Python | MIT |
| [AgentOps](https://github.com/AgentOps-AI/agentops) | Python SDK for AI agent monitoring, LLM cost tracking, benchmarking | Python | MIT |
| [Swarm](https://github.com/openai/swarm) | Educational framework for exploring ergonomic, lightweight multi-agent orchestration | Python | MIT | 
| [Agency Swarm](https://github.com/VRSEN/agency-swarm) | An open-source agent framework designed to automate your workflows | Python | MIT | 
| [Multi-Agent Systems](https://github.com/microsoft/multi-agent-systems) | Research into multi-agent systems and applications | Python | MIT | 
| [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) | Autonomous AI agent for task execution using GPT models | Python | MIT |
| [BabyAGI](https://github.com/yoheinakajima/babyagi) | Task-driven autonomous agent inspired by AGI | Python | MIT |
| [SuperAGI](https://github.com/TransformerOptimus/SuperAGI) | Infrastructure for building and managing autonomous agents | Python | MIT |
| [Griptape](https://github.com/griptape-ai/griptape) | Framework for building AI agents with structured pipelines and memory | Python | Apache-2.0 |
| [Letta (MemGPT)](https://github.com/LettaAI/memgpt) | Long-term memory management for LLM agents | Python | MIT |
| [Agno](https://github.com/agno-ai/agno) | Framework for building AI agents with RAG, workflows, and memory | Python | Apache-2.0 |
| [Agents SDK](https://github.com/vercel/ai) | SDK from Vercel for building agentic workflows and applications | TypeScript | Apache-2.0 |
| [Smolagents](https://github.com/huggingface/smolagents) | Lightweight agent framework from Hugging Face | Python | Apache-2.0 |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Agent framework built on Pydantic for structured reasoning | Python | MIT |
| [CAMEL](https://github.com/camel-ai/camel) | Multi-agent framework enabling role-play and collaboration | Python | Apache-2.0 |
| [Swarms](https://github.com/kyegomez/swarms) | Enterprise agent orchestration framework (“Agency Swarm”) | Python | MIT |
| [Langroid](https://github.com/langroid/langroid) | Framework for building multi-agent conversational systems | Python | Apache-2.0 |
| [Upsonic](https://github.com/upsonic/upsonic) | Agent framework focused on context management and tool use | Python | Apache-2.0 |
| [Agents SDK](https://platform.openai.com/docs/guides/agents-sdk) | OpenAI's official SDK for building agentic apps with context and tool handoff | Python | MIT |
| [BeeAI](https://github.com/i-am-bee/beeai-framework) | Framework for building production-ready multi-agent systems in Python | Python | Apache-2.0 |
| [Composio](https://github.com/ComposioHQ/composio) | Production-ready toolset for connecting AI agents to external services | Python | Apache-2.0 |
| [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents) | Framework for building AI agents using an atomic design philosophy | Python | MIT |
| [OpenWebAgent](https://github.com/THUDM/OpenWebAgent) | Toolkit to enable web-based agents on large language models | Python | Apache-2.0 |
| [Lagent](https://github.com/InternLM/lagent) | Lightweight framework for building LLM-based agents | Python | Apache-2.0 |
| [LazyLLM](https://github.com/LazyAGI/LazyLLM) | Low-code development tool for building multi-agent applications | Python | Apache-2.0 |
| [ChatArena](https://github.com/Farama-Foundation/chatarena) | Library for multi-agent language game environments and social research | Python | Apache-2.0 |
| [Agentarium](https://github.com/Thytu/Agentarium) | Framework for creating simulations populated with AI-powered agents | Python | MIT |
| [Flow](https://github.com/lmnr-ai/flow) | Lightweight task engine for building and visualizing AI agents | Python | Apache-2.0 |
| [Archgw](https://github.com/katanemo/archgw) | Intelligent gateway for orchestrating and managing agents | Go | Apache-2.0 |

## 📦 LLM Development & Optimization

### LLM Training and Fine-Tuning

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) | High-level PyTorch interface for LLMs | Python | Apache-2.0 | 
| [unsloth](https://github.com/unslothai/unsloth) | Fine-tune LLMs faster with less memory | Python | Apache-2.0 |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Post-training pipeline for AI models | Python | Apache-2.0 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Easy & efficient LLM fine-tuning | Python | Apache-2.0 |
| [PEFT](https://github.com/huggingface/peft) | Parameter-Efficient Fine-Tuning library | Python | Apache-2.0 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Distributed training & inference optimization | Python | MIT | 
| [TRL](https://github.com/huggingface/trl) | Train transformer LMs with reinforcement learning | Python | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Pretrained models for text, vision, and audio tasks | Python | Apache-2.0 |
| [LitGPT](https://github.com/Lightning-AI/LitGPT) | Train and fine-tune LLMs lightning fast | Python | Apache-2.0 |
| [Ludwig](https://github.com/ludwig-ai/ludwig) | Low-code framework for custom LLMs | Python | Apache-2.0 |
| [xTuring](https://github.com/stochasticai/xTuring) | Fast fine-tuning of open-source LLMs | Python | Apache-2.0 |
| [RL4LMs](https://github.com/allenai/RL4LMs) | RL library to fine-tune LMs to human preferences | Python | Apache-2.0 |
| [torchtune](https://github.com/pytorch/torchtune) | PyTorch-native library for fine-tuning LLMs | Python | BSD-3 |
| [Accelerate](https://github.com/huggingface/accelerate) | Library to easily train on multiple GPUs/TPUs with mixed precision | Python | Apache-2.0 |
| [LLMBox](https://github.com/RUCAIBox/LLMBox) | Comprehensive library for implementing LLMs with unified training and evaluation pipelines | Python | MIT |
| [Mergoo](https://github.com/Leeroo-AI/mergoo) | Library for easily merging multiple LLM experts and efficiently training the result | Python | MIT |
| [Txtinstruct](https://github.com/neuml/txtinstruct) | A framework for training instruction-tuned models from various data sources | Python | Apache-2.0 |
| [Lamini](https://github.com/lamini-ai/lamini) | An integrated LLM inference and tuning platform for enterprise use | Python | Apache-2.0 |
| [torchchat](https://github.com/pytorch/torchchat) | Run PyTorch LLMs locally on servers, desktops, and mobile devices | Python | BSD-3-Clause |

### Open Source LLM Inference

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LLM Compressor](https://github.com/mit-han-lab/llm-compressor) | Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment | Python | Apache-2.0 |
| [LightLLM](https://github.com/ModelTC/lightllm) | Lightweight Python-based LLM inference and serving framework with easy scalability and high performance | Python | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput and memory-efficient inference and serving engine for LLMs | Python | Apache-2.0 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA library for optimizing LLM inference with TensorRT | C++/Python | Apache-2.0 |
| [WebLLM](https://github.com/mlc-ai/web-llm) | High-performance in-browser LLM inference engine | TypeScript/Python | Apache-2.0 |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Unified framework to run ML workloads and LLMs on any cloud (AWS, GCP, Azure, Lambda, etc.) with auto-spot, data syncing, and cost optimization. | Python | Apache-2.0 |

### LLM Safety and Security

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Guardrails](https://github.com/ShreyaR/guardrails) | Add guardrails to large language models | Python | MIT |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Toolkit for adding programmable guardrails to LLM conversational systems | Python | Apache-2.0 |
| [Garak](https://github.com/leondz/garak) | LLM vulnerability scanner | Python | MIT |
| [DeepTeam](https://github.com/DeepTeamAI/deepteam) | LLM red teaming framework | Python | Apache-2.0 |

### AI App Development Frameworks

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Reflex](https://github.com/reflex-dev/reflex) | Build full-stack web apps powered by LLMs with Python-only workflows and reactive UIs. | Python | Apache-2.0 |
| [Gradio](https://github.com/gradio-app/gradio) | Create quick, interactive UIs for LLM demos and prototypes. | Python | Apache-2.0 |
| [Streamlit](https://github.com/streamlit/streamlit) | Build and share AI/ML apps fast with Python scripts and interactive widgets. | Python | Apache-2.0 |
| [Taipy](https://github.com/Avaiga/taipy) | End-to-end Python framework for building production-ready AI apps with dashboards and pipelines. | Python | Apache-2.0 |
| [AI SDK UI](https://github.com/vercel/ai) | Vercel’s AI SDK for building chat & generative UIs | TypeScript | Apache-2.0 |
| [Simpleaichat](https://github.com/minimaxir/simpleaichat) | Minimal Python interface for prototyping conversational LLMs | Python | MIT |
| [Chainlit](https://github.com/Chainlit/chainlit) | Framework for building and debugging LLM apps with a rich UI | Python | Apache-2.0 |

### Local Development & Serving

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Ollama](https://github.com/ollama/ollama) | Get up and running with large language models locally | Go | MIT |
| [LM Studio](https://lmstudio.ai/) | Desktop app for running local LLMs | - | Proprietary |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | Open-source chatbot ecosystem | C++ | MIT |
| [LocalAI](https://github.com/mudler/LocalAI) | Self-hosted OpenAI-compatible API | Go | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Lightweight, high-performance inference engine for running LLMs locally across CPU, GPU, and mobile backends | C++ | MIT |
| [LiteLLM](https://github.com/BerriAI/litellm) | Lightweight OpenAI-compatible gateway for multiple LLM providers | Python | MIT |
| [AI Gateway](https://github.com/Portkey-AI/ai-gateway) | Gateway for managing LLM requests, caching, and routing | Python | Apache-2.0 |
| [Langcorn](https://github.com/langcorn/langcorn) | Serve LangChain applications via FastAPI with production-ready endpoints | Python | MIT |
| [LitServe](https://github.com/Lightning-AI/LitServe) | High-speed GPU inference server with autoscaling and batch support | Python | Apache-2.0 |

### LLM Inference Platforms

| Platform | Description | Pricing | Features |
|----------|-------------|---------|----------|
| [Clarifai](https://www.clarifai.com/) | Lightning-fast compute for AI models & agents | Free tier + Pay-as-you-go | Pre-trained models, Deploy your own models on Dedicated compute, Model training, Workflow automation | 
| [Modal](https://modal.com/) | Serverless platform for AI/ML workloads | Pay-per-use | Serverless GPU, Auto-scaling |
| [Replicate](https://replicate.com/) | Run open-source models with a cloud API | Pay-per-use | Pre-built models, Custom training |
| [Together AI](https://www.together.ai/) | Cloud platform for open-source models | Various | Open models, Fine-tuning |
| [Anyscale](https://www.anyscale.com/) | Ray-based platform for AI applications | Enterprise | Distributed training, Serving |
| [Groq](https://groq.com/) | Ultra-fast LPU inference engine for running open AI models |Free tier + Pay-per-use | Blazing fast speed, High throughput, Low latency, Open model support |
| [OpenRouter](https://openrouter.ai/) | Universal API to find and route to the best LLMs from various providers | Free tier + Pay-per-use | Multi-provider access, Unified API, Model comparison, Caching |
| [RouteLLM](https://github.com/routeLLM/routeLLM) | Dynamic router for selecting best LLMs based on cost & performance | Open-source | Cost optimization, Multi-LLM routing |

### Structured Output & Constraints

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Instructor](https://github.com/instructor-ai/instructor) | Python library for working with structured outputs using Pydantic | Python | MIT |
| [XGrammar](https://github.com/mlc-ai/xgrammar) | Open-source library for efficient and portable structured generation | C++/Python | Apache-2.0 |
| [Outlines](https://github.com/dottxt-ai/outlines) | Robust library for structured text generation and constraint guided output | Python | Apache-2.0 |
| [Guidance](https://github.com/guidance-ai/guidance) | Programming paradigm for steering and controlling language models | Python | MIT |
| [LMQL](https://github.com/eth-sri/lmql) | Language for constraint-guided and efficient LLM programming | Python/JS | MIT/Apache |
| [Jsonformer](https://github.com/1rgs/jsonformer) | A way to generate structured JSON from language models consistently | Python | MIT |

### Prompt Optimization & Compression

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [PCToolkit](https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression) | Unified plug-and-play prompt compression toolkit for LLMs | Python | Apache-2.0 |
| [Selective Context](https://pypi.org/project/selective-context/) | Compresses prompts and context to allow LLMs to process more content | Python | MIT |
| [LLMLingua](https://github.com/microsoft/LLMLingua) | Library for compressing prompts to accelerate LLM inference | Python | MIT |
| [betterprompt](https://github.com/stjordanis/betterprompt) | Test suite for LLM prompts before deploying to production | Python | MIT |
| [Promptify](https://github.com/promptslab/Promptify) | Tool for easily generating prompts for various NLP tasks | Python | Apache-2.0 |
| [PromptSource](https://pypi.org/project/promptsource/) | Toolkit for creating, sharing, and using natural language prompts | Python | Apache-2.0 |
| [Py-priompt](https://github.com/zenbase-ai/py-priompt) | Advanced prompt design and management library | Python | Apache-2.0 |
| [Promptimizer](https://github.com/hinthornw/promptimizer) | Library designed for automatic prompt optimization | Python | MIT |

### Knowledge Editing & Reasoning

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [LLM Reasoners](https://github.com/maitrix-org/llm-reasoners) | Library specifically for advanced large language model reasoning | Python | MIT |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | Easy-to-use knowledge editing framework for large language models | Python | Apache-2.0 |
| [LLM Reasoner](https://github.com/harishsg993010/LLM-Reasoner) | Library to make any LLM exhibit reasoning like OpenAI o1 | Python | MIT |
| [Vanna](https://github.com/vanna-ai/vanna) | Accurate Text-to-SQL generation using RAG and LLMs | Python | MIT |
| [mergekit](https://github.com/arcee-ai/MergeKit) | Tools for merging pretrained large language models effectively | Python | MIT/Apache |
| [MarkLLM](https://github.com/THU-BPM/MarkLLM) | Open-source toolkit for LLM text watermarking | Python | Apache-2.0 |
| [LLMSanitize](https://github.com/ntunlp/LLMSanitize) | Library for contamination detection in NLP datasets and LLMs | Python | Apache-2.0 |


## 🤝 Contributing

We welcome contributions! This toolkit grows stronger with community input.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-tool`)
3. **Add your contribution** (new tool, template, or tutorial)
4. **Submit a pull request**

### Contribution Guidelines

- **Quality over quantity** - Focus on tools and resources that provide real value
- **Production-ready** - Include tools that work in real-world scenarios
- **Well-documented** - Provide clear descriptions and usage examples
- **Up-to-date** - Ensure tools are actively maintained

---

## 📧 Stay Connected

### Newsletter
Get weekly AI engineering insights, tool reviews, and exclusive demos and AI Projects delivered to your inbox:

**[📧 Subscribe to AI Engineering Newsletter →](https://aiengineering.beehiiv.com/subscribe)**

*Join 100,000+ engineers building better LLM applications*

### Social Media
[![X Follow](https://img.shields.io/twitter/follow/Sumanth_077?style=social&logo=x)](https://x.com/Sumanth_077)
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-Follow-blue?style=social&logo=linkedin)](https://www.linkedin.com/company/theaiengineering/)

---

**Built with ❤️ for the AI Engineering community**

*Star ⭐ this repo if you find it helpful!*
