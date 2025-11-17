# AI Engineering Toolkit🔥

**Build better LLM apps — faster, smarter, production-ready.**

A curated, list of 100+ libraries and frameworks for AI engineers building with Large Language Models. This toolkit includes battle-tested tools, frameworks, templates, and reference implementations for developing, deploying, and optimizing LLM-powered systems.

[![Toolkit banner](https://github.com/codedspaces/demo-2/blob/d9442b179eba2856e8c6e62bb1c6a1bb8c676b89/2.jpg?raw=true)](https://aiengineering.beehiiv.com/subscribe)

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

## 📋 Table of Contents

- [🛠️ Tooling for AI Engineers](#%EF%B8%8F-tooling-for-ai-engineers)
  - [Vector Databases](#vector-databases)
  - [Orchestration & Workflows](#orchestration--workflows)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Evaluation & Testing](#evaluation--testing)
  - [Model Management](#model-management)
  - [Data Collection & Web Scraping](#data-collection--web-scraping)
- [🤖 Agent Frameworks](#-agent-frameworks)
- [📦 LLM Development & Optimization](#llm-development--optimization)
  - [Open Source LLM Inference](#open-source-llm-inference)
  - [LLM Safety & Security](#llm-safety--security)
  - [AI App Development Frameworks](#ai-app-development-frameworks)
  - [Local Development & Serving](#local-development--serving)
  - [LLM Inference Platforms](#llm-inference-platforms)
- [🤝 Contributing](#-contributing)

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
| [Chonkie](https://github.com/bhavnicksm/chonkie-main) | Chunking utility for efficient document processing in RAG | Python | - |
| [SQLite-Vec](https://github.com/asg017/sqlite-vec) | Vector search extension for SQLite, useful in lightweight RAG setups | C/Python | MIT |
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Low-latency RAG research toolkit with modular design and benchmarks | Python | - |
| [Llmware](https://github.com/llmware-ai/llmware) | Lightweight framework for building RAG-based apps | Python | Apache-2.0 |
| [Vectara](https://github.com/vectara) | Managed RAG platform with APIs for retrieval and generation | Python/Go | Commercial |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Semantic cache for LLM responses to accelerate RAG pipelines | Python | Apache-2.0 |

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
| [LM Studio](https://lmstudio.ai/) | Desktop app for running local LLMs | - | Commercial |
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
