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
  - [MCP Tools & Servers](#mcp-tools--servers)
  - [Voice AI & Speech Tools](#voice-ai--speech-tools)
- [🤖 Agent Frameworks](#-agent-frameworks)
- [📦 LLM Development & Optimization](#-llm-development--optimization)
  - [Open Source LLM Inference](#open-source-llm-inference)
  - [LLM Safety & Security](#llm-safety-and-security)
  - [AI App Development Frameworks](#ai-app-development-frameworks)
  - [Local Development & Serving](#local-development--serving)
  - [LLM Inference Platforms](#llm-inference-platforms)
  - [Structured Generation](#structured-generation)
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
| [Vectara](https://github.com/vectara) | Managed RAG platform with APIs for retrieval and generation | Python/Go | Commercial |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity search extension for PostgreSQL | C | PostgreSQL License |
| [LanceDB](https://github.com/lancedb/lancedb) | Serverless embedded vector database for multimodal AI built on the Lance columnar format | Rust/Python | Apache-2.0 |
| [Vespa](https://github.com/vespa-engine/vespa) | Open-source platform for large-scale vector search, ranking, and real-time recommendation | Java/C++ | Apache-2.0 |
| [OpenSearch](https://github.com/opensearch-project/OpenSearch) | Distributed search and analytics engine with vector search and hybrid search support | Java | Apache-2.0 |

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
| [Dify](https://github.com/langgenius/dify) | Production-ready open source platform combining RAG pipelines, agent capabilities, model management, and observability in one UI | TypeScript/Python | Apache-2.0 |
| [n8n](https://github.com/n8n-io/n8n) | Workflow automation platform with native LLM and AI agent node support for building agentic pipelines | TypeScript | Sustainable Use |
| [Instructor](https://github.com/jxnl/instructor) | Structured outputs from LLMs using Pydantic — reliable JSON extraction with validation and retries | Python | MIT |
| [Outlines](https://github.com/dottxt-ai/outlines) | Structured text generation with guaranteed JSON schema compliance and regex-guided sampling | Python | Apache-2.0 |
| [Pathway](https://github.com/pathwaycom/pathway) | Python framework for real-time data pipelines connecting live data streams to LLM and RAG workflows | Python | BSL |

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
| [MarkItDown](https://github.com/microsoft/markitdown) | Microsoft tool for converting files and office documents to Markdown for use with LLMs | Python | MIT |

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
| [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Low-latency RAG research toolkit with modular design and benchmarks | Python | - |
| [Llmware](https://github.com/llmware-ai/llmware) | Lightweight framework for building RAG-based apps | Python | Apache-2.0 |
| [LightRAG](https://github.com/HKUDS/LightRAG) | Graph-based RAG with knowledge graph extraction, multimodal support, and multiple chunking strategies | Python | MIT |
| [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) | Easily use and train ColBERT-based late interaction retrieval models in any RAG pipeline | Python | Apache-2.0 |
| [Cognita](https://github.com/truefoundry/cognita) | Modular, production-ready RAG framework with UI, API layer, and support for multiple vector DBs | Python | Apache-2.0 |
| [Kotaemon](https://github.com/Cinnamon/kotaemon) | Open-source RAG-based document QA tool with clean UI, hybrid retrieval, and multi-modal document support | Python | Apache-2.0 |
| [Memvid](https://github.com/memvid/memvid) | Single-file memory layer for AI agents — replaces complex RAG pipelines with serverless, portable memory | Rust | Apache-2.0 |

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
| [Langfuse](https://github.com/langfuse/langfuse) | Open source LLM engineering platform with tracing, evals, prompt management, and metrics | TypeScript/Python | MIT |
| [Helicone](https://github.com/Helicone/helicone) | Open source observability and monitoring platform for debugging and improving LLM apps | TypeScript | Apache-2.0 |
| [OpenLLMetry](https://github.com/traceloop/openllmetry) | OpenTelemetry-based observability extensions for LLM apps — plugs into Datadog, Honeycomb, and more | Python | Apache-2.0 |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | Test and evaluate LLM outputs with assertions, red teaming, and CI/CD integration | TypeScript | MIT |
| [Braintrust](https://github.com/braintrustdata/braintrust-sdk-javascript) | End-to-end AI evaluation platform connecting production traces to systematic improvement and CI/CD | TypeScript | MIT |
| [AgentBench](https://github.com/THUDM/AgentBench) | Comprehensive benchmark for evaluating LLMs as autonomous agents across diverse real-world environments | Python | Apache-2.0 |

### Model Management

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Client library for Hugging Face Hub | Python | Apache-2.0 | 
| [MLflow](https://github.com/mlflow/mlflow) | Platform for ML lifecycle management | Python | Apache-2.0 |
| [Weights & Biases](https://github.com/wandb/wandb) | Developer tools for ML | Python | MIT |
| [DVC](https://github.com/iterative/dvc) | Data version control for ML projects | Python | Apache-2.0 |
| [ClearML](https://github.com/allegroai/clearml) | End-to-end MLOps platform with LLM support | Python | Apache-2.0 |
| [ZenML](https://github.com/zenml-io/zenml) | MLOps framework for building pipelines and managing models across any cloud infrastructure | Python | Apache-2.0 |
| [BentoML](https://github.com/bentoml/BentoML) | Unified model serving framework for building and deploying AI model inference APIs | Python | Apache-2.0 |
| [Langfuse](https://github.com/langfuse/langfuse) | Open-source LLM engineering platform for tracing, evaluation, prompt management, and observability | TypeScript/Python | MIT |

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
| [Scrapling](https://github.com/D4Vinci/Scrapling) | Adaptive web scraping framework with stealth mode, anti-bot bypass, and built-in MCP server support | Python | BSD-3 |


### MCP Tools & Servers

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Official Python SDK for building MCP servers and clients — the standard way to expose tools and data to LLMs | Python | MIT |
| [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) | Official TypeScript SDK for building MCP servers and clients with full protocol support | TypeScript | MIT |
| [MCP Servers](https://github.com/modelcontextprotocol/servers) | Official reference MCP server implementations including Filesystem, Memory, Git, Fetch, and Sequential Thinking | TypeScript/Python | MIT |
| [Context7](https://github.com/upstash/context7) | Injects live, version-specific library documentation into LLM prompts — eliminates hallucinated APIs | TypeScript | MIT |
| [Playwright MCP](https://github.com/microsoft/playwright-mcp) | Official Microsoft MCP server for browser automation using Playwright with accessibility-based snapshots | TypeScript | Apache-2.0 |
| [Smithery CLI](https://github.com/smithery-ai/cli) | CLI for discovering, installing, and managing MCP servers across AI development environments | TypeScript | AGPL-3.0 |
| [FastMCP](https://github.com/PrefectHQ/fastmcp) | High-level Python framework for building MCP servers — minimal boilerplate, production-ready | Python | MIT |
| [mcp-use](https://github.com/mcp-use/mcp-use) | Fullstack MCP framework for building MCP servers, apps, and agents with TypeScript and Python support | TypeScript/Python | MIT |
| [ToolHive](https://github.com/stacklok/toolhive) | CLI utility for securely deploying and managing MCP servers using containerization and standardized packaging | Go | Apache-2.0 |
| [Cognee](https://github.com/topoteretes/cognee) | Memory manager for AI agents using graph and vector stores — connects to 30+ data sources with MCP support | Python | Apache-2.0 |


### Voice AI & Speech Tools

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Pipecat](https://github.com/pipecat-ai/pipecat) | Open-source Python framework for building real-time voice and multimodal AI agents with STT, LLM, and TTS pipeline orchestration | Python | BSD-2-Clause |
| [LiveKit Agents](https://github.com/livekit/agents) | Production-grade framework for building real-time voice AI agents with WebRTC transport, telephony support, and native MCP integration | Python | Apache-2.0 |
| [Whisper](https://github.com/openai/whisper) | OpenAI's general-purpose speech recognition model trained on 680k hours of multilingual audio | Python | MIT |
| [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) | High-performance C/C++ port of Whisper for fast local inference across CPU, GPU, and edge devices | C/C++ | MIT |
| [Kokoro](https://github.com/hexgrad/kokoro) | Lightweight open-weight TTS model with 82M parameters — fast, high-quality speech synthesis with Apache-licensed weights | Python | Apache-2.0 |
| [Coqui TTS](https://github.com/idiap/coqui-ai-TTS) | Deep learning toolkit for text-to-speech with 1000+ pre-trained models, voice cloning, and multilingual support | Python | MPL-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Lightweight, enterprise-grade voice activity detection model — zero telemetry, no vendor lock, runs on CPU | Python | MIT |


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
| [Mem0](https://github.com/mem0ai/mem0) | Universal memory layer for AI agents — persistent, personalized memory across sessions | Python | Apache-2.0 |
| [Mastra](https://github.com/mastra-ai/mastra) | TypeScript-native agent framework with workflow-driven development and built-in observability | TypeScript | Apache-2.0 |
| [MetaGPT](https://github.com/FoundationAgents/MetaGPT) | Multi-agent framework that simulates a software company with role-playing agents (PM, architect, engineer) | Python | MIT |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | Production-ready multi-agent framework with native MCP tool use and sub-agent handoffs | Python | MIT |
| [AG2](https://github.com/ag2ai/ag2) | Actively maintained evolution of AutoGen by the original authors with enhanced multi-agent capabilities | Python | Apache-2.0 |
| [OpenClaw](https://github.com/openclaw/openclaw) | Self-hosted autonomous AI agent connecting to 10+ messaging platforms with persistent memory and 24/7 operation | TypeScript | MIT |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Self-improving autonomous agent by Nous Research with skill creation, memory, and 18-platform messaging gateway | Python | MIT |
| [DeerFlow](https://github.com/bytedance/deer-flow) | ByteDance's open-source SuperAgent harness orchestrating sub-agents, memory, and sandboxes for long-horizon tasks | Python/TypeScript | MIT |
| [AgenticSeek](https://github.com/Fosowl/agenticSeek) | Fully local autonomous agent that thinks, browses the web, and writes code — no APIs required | Python | GPL-3.0 |

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
| [LMFlow](https://github.com/OptimalScale/LMFlow) | Extensible toolkit for finetuning and inference of large foundation models | Python | Apache-2.0 |
| [MergeKit](https://github.com/arcee-ai/mergekit) | Toolkit for merging multiple fine-tuned LLMs into a single model without additional training | Python | LGPL-3.0 |
| [ART](https://github.com/OpenPipe/ART) | Agent Reinforcement Trainer — train multi-step agents for real-world tasks using GRPO | Python | Apache-2.0 |

### Open Source LLM Inference

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LLM Compressor](https://github.com/mit-han-lab/llm-compressor) | Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment | Python | Apache-2.0 |
| [LightLLM](https://github.com/ModelTC/lightllm) | Lightweight Python-based LLM inference and serving framework with easy scalability and high performance | Python | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput and memory-efficient inference and serving engine for LLMs | Python | Apache-2.0 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA library for optimizing LLM inference with TensorRT | C++/Python | Apache-2.0 |
| [WebLLM](https://github.com/mlc-ai/web-llm) | High-performance in-browser LLM inference engine | TypeScript/Python | Apache-2.0 |
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Unified framework to run ML workloads and LLMs on any cloud (AWS, GCP, Azure, Lambda, etc.) with auto-spot, data syncing, and cost optimization. | Python | Apache-2.0 |
| [Gemma](https://github.com/google-deepmind/gemma) | Google DeepMind's open-weight LLM library — supports Gemma 4 with multimodal input including text, images, video, and audio | Python | Apache-2.0 |

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
| [Mirascope](https://github.com/mirascope/mirascope) | Lightweight Python toolkit for building LLM apps with Pydantic-native structured outputs and no extra abstractions | Python | MIT |

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
| [Jan](https://github.com/janhq/jan) | Open-source ChatGPT alternative that runs 100% offline with local model management | TypeScript | AGPL-3.0 |
| [Llamafile](https://github.com/Mozilla-Ocho/llamafile) | Distribute and run LLMs as a single self-contained executable file — zero setup required | C++ | Apache-2.0 |

### Structured Generation

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Instructor](https://github.com/567-labs/instructor) | Multi-language library for extracting structured, validated outputs from LLMs using Pydantic | Python/TS/Go | MIT |
| [Outlines](https://github.com/dottxt-ai/outlines) | Guarantees 100% valid structured outputs (JSON, regex, grammars) by constraining LLM token generation | Python | Apache-2.0 |
| [Guidance](https://github.com/guidance-ai/guidance) | Microsoft's language for controlling LLM outputs — interleave generation with conditionals, loops, and constraints | Python | MIT |

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
| [Fireworks AI](https://fireworks.ai/) | High-performance inference platform optimized for agentic systems and structured output | Free credits + Pay-as-you-go | Ultra-low latency, Structured output, Fine-tuning, 50+ open models |
| [Cerebras](https://www.cerebras.ai/inference) | Wafer-scale chip inference platform delivering the world's fastest LLM throughput | Free tier + Pay-per-use | World's fastest inference, OpenAI-compatible API, Function calling, JSON mode |
| [Hyperbolic](https://www.hyperbolic.ai/) | Open-access AI cloud for lowest-cost inference and GPU rentals on open-source models | Pay-per-use | Lowest-cost inference, GPU rentals, OpenAI-compatible, No hidden fees |
| [Hugging Face Inference](https://huggingface.co/inference) | Unified API routing to 15+ inference providers with access to 200+ open-source models | Free tier + Pay-as-you-go | 200+ models, 15+ providers, Serverless and dedicated endpoints, Monthly free credits |

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
