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
  - [Prompt Engineering & Optimization](#prompt-engineering--optimization)
  - [Structured Output & Constrained Generation](#structured-output--constrained-generation)
- [🤖 Agent Frameworks](#-agent-frameworks)
- [📦 LLM Development & Optimization](#llm-development--optimization)
  - [Open Source LLM Inference](#open-source-llm-inference)
  - [LLM Safety & Security](#llm-safety--security)
  - [AI App Development Frameworks](#ai-app-development-frameworks)
  - [Local Development & Serving](#local-development--serving)
  - [LLM Data Generation](#llm-data-generation)
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
| [Ragas](https://github.com/explodinggradients/ragas) | Evaluation framework for RAG pipelines | Python | Apache-2.0 |
| [LangSmith](https://smith.langchain.com/) | Platform for debugging, testing, and monitoring LLM applications | API/SDK | Commercial |
| [Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLM, vision, language, and tabular models | Python | Apache-2.0 |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation framework for unit testing LLM outputs | Python | Apache-2.0 |
| [TruLens](https://github.com/truera/trulens) | Evaluation and tracking for LLM experiments | Python | MIT |
| [Inspect](https://github.com/ukaisi/inspect) | Framework for large language model evaluations | Python | Apache-2.0 |
| [UpTrain](https://github.com/uptrain-ai/uptrain) | Open-source tool to evaluate and improve LLM applications | Python | Apache-2.0 |
| [Weave](https://github.com/wandb/weave) | Experiment tracking, debugging, and logging for LLM workflows | Python | Apache-2.0 |
| [Giskard](https://github.com/Giskard-AI/giskard) | Open-source testing framework for ML/LLM applications | Python | Apache-2.0 |
| [Lighteval](https://github.com/huggingface/lighteval) | Lightweight and fast evaluation framework from Hugging Face | Python | Apache-2.0 |
| [LangTest](https://github.com/JohnSnowLabs/langtest) | NLP/LLM test suite for robustness, bias, and quality | Python | Apache-2.0 |
| [PromptBench](https://github.com/microsoft/promptbench) | Benchmarking framework for evaluating prompts | Python | MIT |
| [EvalPlus](https://github.com/evalplus/evalplus) | Advanced evaluation framework for code generation models | Python | Apache-2.0 |
| [FastChat](https://github.com/lm-sys/FastChat) | Framework for chat-based LLM benchmarking and evaluation | Python | Apache-2.0 |
| [judges](https://github.com/stanford-crfm/judges) | Human + AI judging framework for LLM evaluation | Python | Apache-2.0 |
| [Evals](https://github.com/openai/evals) | OpenAI's framework for creating and running LLM evaluations | Python | MIT |
| [AgentEvals](https://github.com/agent-evals/agent-evals) | Evaluation framework for autonomous AI agents | Python | Apache-2.0 |
| [UQLM](https://github.com/uqfoundation/uqlm) | Unified framework for evaluating quality of LLMs | Python | Apache-2.0 |
| [LLMBox](https://github.com/llmbox/llmbox) | Toolkit for evaluation + training of LLMs | Python | Apache-2.0 |
| [Opik](https://github.com/opik-ai/opik) | DevOps platform for evaluation, monitoring, and observability | Python | Apache-2.0 |
| [PydanticAI Evals](https://github.com/pydantic/pydantic-ai) | Built-in evaluation utilities for PydanticAI agents | Python | MIT |
| [LLM Transparency Tool](https://github.com/transparency-ai/llm-transparency-tool) | Framework for probing and evaluating LLM transparency | Python | Apache-2.0 |
| [AnnotateAI](https://github.com/annotate-ai/annotate-ai) | Annotation and evaluation framework for LLM datasets | Python | Apache-2.0 |

### Model Management

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Client library for Hugging Face Hub | Python | Apache-2.0 | 
| [MLflow](https://github.com/mlflow/mlflow) | Platform for ML lifecycle management | Python | Apache-2.0 |
| [Weights & Biases](https://github.com/wandb/wandb) | Developer tools for ML | Python | MIT |
| [DVC](https://github.com/iterative/dvc) | Data version control for ML projects | Python | Apache-2.0 |
| [Comet ML](https://github.com/comet-ml/comet-ml) | Experiment tracking and visualization for ML/LLM workflows | Python | MIT |
| [ClearML](https://github.com/allegroai/clearml) | End-to-end MLOps platform with LLM support | Python | Apache-2.0 |

### Data Collection & Web Scraping

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Firecrawl](https://github.com/mendableai/firecrawl) | AI-powered web crawler that extracts and structures content for LLM pipelines | TypeScript | MIT |
| [Scrapy](https://github.com/scrapy/scrapy) | Fast, high-level web crawling & scraping framework | Python | BSD-3 |
| [Playwright](https://github.com/microsoft/playwright) | Web automation & scraping with headless browsers | TypeScript/Python/Java/.NET | Apache-2.0 | 
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) | Easy HTML/XML parsing for quick scraping tasks | Python | MIT |
| [Selenium](https://github.com/SeleniumHQ/selenium) | Browser automation framework (supports scraping) | Multiple | Apache-2.0 |
| [Apify SDK](https://github.com/apify/apify-sdk-python) | Web scraping & automation platform SDK | Python/JavaScript | Apache-2.0 |
| [Newspaper3k](https://github.com/codelucas/newspaper) | News & article extraction library | Python | MIT |
| [Data Prep Kit](https://github.com/databricks/data-prep-kit) | Toolkit for cleaning, transforming, and preparing datasets for LLMs | Python | Apache-2.0 |
| [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai) | Use LLMs to extract structured data from websites and documents | Python | MIT |
| [Crawlee](https://github.com/apify/crawlee) | Web scraping and crawling framework for large-scale data collection | TypeScript | Apache-2.0 |

### Prompt Engineering & Optimization

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Promptify](https://github.com/promptslab/Promptify) | Prompt engineering toolkit for NLP/LLM tasks | Python | Apache-2.0 |
| [PromptSource](https://github.com/bigscience-workshop/promptsource) | Toolkit for creating, sharing, and managing prompts | Python | Apache-2.0 |
| [Promptimizer](https://github.com/microsoft/promptimizer) | Microsoft toolkit for optimizing prompts via evaluation | Python | MIT |
| [Py-Priompt](https://github.com/py-priompt/py-priompt) | Library for prioritizing and optimizing LLM prompts | Python | MIT |
| [Selective Context](https://github.com/microsoft/selective-context) | Context selection and compression for efficient prompting | Python | MIT |
| [LLMLingua](https://github.com/microsoft/LLMLingua) | Prompt compression via token selection and ranking | Python | MIT |
| [betterprompt](https://github.com/jxnl/betterprompt) | Prompt experimentation & optimization framework | Python | Apache-2.0 |
| [PCToolkit](https://github.com/PCToolkit/pc-toolkit) | Toolkit for prompt compression and efficiency | Python | Apache-2.0 |

### Structured Output & Constrained Generation

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Instructor](https://github.com/jxnl/instructor) | Structured LLM outputs with Pydantic schema validation | Python | MIT |
| [XGrammar](https://github.com/bojone/XGrammar) | Grammar-based constrained generation for LLMs | Python | Apache-2.0 |
| [Outlines](https://github.com/outlines-dev/outlines) | Controlled generation with regex, CFGs, and schemas | Python | MIT |
| [Guidance](https://github.com/guidance-ai/guidance) | Programmatic control of LLM outputs with constraints | Python | MIT |
| [LMQL](https://github.com/eth-sri/lmql) | Query language for structured interaction with LLMs | Python | Apache-2.0 |
| [Jsonformer](https://github.com/1rgs/jsonformer) | Efficient constrained decoding for valid JSON outputs | Python | MIT |

## 🤖 Agent Frameworks

| Framework | Description | Language | License |
|-----------|-------------|----------|---------|
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
| [Phidata](https://github.com/phidatahq/phidata) | Build AI agents with memory, tools, and knowledge | Python | MIT |
| [MemGPT](https://github.com/cpacker/MemGPT) | Self-improving agents with infinite context via memory management | Python | MIT |
| [Griptape](https://github.com/griptape-ai/griptape) | Framework for building AI agents with structured pipelines and memory | Python | Apache-2.0 |
| [mem0](https://github.com/mem0ai/mem0) | AI memory framework for storing & retrieving agent context across sessions | Python | MIT |
| [Memoripy](https://github.com/memoripy/memoripy) | Lightweight persistent memory library for LLMs and agents | Python | MIT |
| [Memobase](https://github.com/memobase/memobase) | Database-like persistent memory for conversational agents | Python | MIT |
| [Letta (MemGPT)](https://github.com/LettaAI/memgpt) | Long-term memory management for LLM agents | Python | MIT |
| [Agno](https://github.com/agno-ai/agno) | Framework for building AI agents with RAG, workflows, and memory | Python | Apache-2.0 |
| [Agents SDK](https://github.com/vercel/ai) | SDK from Vercel for building agentic workflows and applications | TypeScript | Apache-2.0 |
| [Smolagents](https://github.com/huggingface/smolagents) | Lightweight agent framework from Hugging Face | Python | Apache-2.0 |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Agent framework built on Pydantic for structured reasoning | Python | MIT |
| [CAMEL](https://github.com/camel-ai/camel) | Multi-agent framework enabling role-play and collaboration | Python | Apache-2.0 |
| [BeeAI](https://github.com/bee-ai/bee-ai) | LLM agent framework for AI-driven workflows and automation | Python | Apache-2.0 |
| [gradio-tools](https://github.com/freddyaboulton/gradio-tools) | Integrate external tools into agents via Gradio apps | Python | Apache-2.0 |
| [Composio](https://github.com/composio/composio) | Tool orchestration framework to connect 100+ APIs for agents | Python | Apache-2.0 |
| [Atomic Agents](https://github.com/atomic-agents/atomic-agents) | Modular agent framework with tool usage and reasoning | Python | Apache-2.0 |
| [Memary](https://github.com/memary-ai/memary) | Memory-augmented agent framework for persistent context | Python | MIT |
| [Browser Use](https://github.com/browser-use/browser-use) | Framework for browser automation with AI agents | Python | Apache-2.0 |
| [OpenWebAgent](https://github.com/open-web-agent/open-web-agent) | Agents for interacting with and extracting from the web | Python | Apache-2.0 |
| [Lagent](https://github.com/InternLM/lagent) | Lightweight agent framework from InternLM | Python | Apache-2.0 |
| [LazyLLM](https://github.com/Lazy-Llm/LazyLLM) | Agent framework for lazy evaluation and efficient execution | Python | Apache-2.0 |
| [Swarms](https://github.com/kyegomez/swarms) | Enterprise agent orchestration framework (“Agency Swarm”) | Python | MIT |
| [ChatArena](https://github.com/chatarena/chatarena) | Multi-agent simulation platform for research and evaluation | Python | Apache-2.0 |
| [AgentStack](https://github.com/agentstack-ai/agentstack) | Agent orchestration framework (different from Agency Swarm) | Python | Apache-2.0 |
| [Archgw](https://github.com/arch-gw/archgw) | Agent runtime for structured workflows and graph execution | Python | Apache-2.0 |
| [Flow](https://github.com/flow-ai/flow) | Low-code agent workflow framework for LLMs | Python | Apache-2.0 |
| [Langroid](https://github.com/langroid/langroid) | Framework for building multi-agent conversational systems | Python | Apache-2.0 |
| [Agentarium](https://github.com/agentarium/agentarium) | Platform for creating multi-agent environments | Python | Apache-2.0 |
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
| [Mergoo](https://github.com/mlfoundations/mergoo) | Merge multiple LLM experts efficiently | Python | Apache-2.0 | 
| [Ludwig](https://github.com/ludwig-ai/ludwig) | Low-code framework for custom LLMs | Python | Apache-2.0 |
| [txtinstruct](https://github.com/allenai/txtinstruct) | Framework for training instruction-tuned models | Python | Apache-2.0 |
| [xTuring](https://github.com/stochasticai/xTuring) | Fast fine-tuning of open-source LLMs | Python | Apache-2.0 |
| [RL4LMs](https://github.com/allenai/RL4LMs) | RL library to fine-tune LMs to human preferences | Python | Apache-2.0 |
| [torchtune](https://github.com/pytorch/torchtune) | PyTorch-native library for fine-tuning LLMs | Python | BSD-3 |
| [Accelerate](https://github.com/huggingface/accelerate) | Library to easily train on multiple GPUs/TPUs with mixed precision | Python | Apache-2.0 |
| [BitsandBytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit optimizers and quantization for efficient LLM training | Python | MIT |
| [Lamini](https://github.com/lamini-ai/lamini) | Python SDK for building and fine-tuning LLMs with Lamini API | Python | Apache-2.0 |

### Open Source LLM Inference

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LLM Compressor](https://github.com/mit-han-lab/llm-compressor) | Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment | Python | Apache-2.0 |
| [LightLLM](https://github.com/ModelTC/lightllm) | Lightweight Python-based LLM inference and serving framework with easy scalability and high performance | Python | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput and memory-efficient inference and serving engine for LLMs | Python | Apache-2.0 |
| [torchchat](https://github.com/facebookresearch/torchchat) | Run PyTorch LLMs locally on servers, desktop, and mobile | Python | MIT |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA library for optimizing LLM inference with TensorRT | C++/Python | Apache-2.0 |
| [WebLLM](https://github.com/mlc-ai/web-llm) | High-performance in-browser LLM inference engine | TypeScript/Python | Apache-2.0 |

### LLM Safety and Security

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [JailbreakEval](https://github.com/centerforaisafety/JailbreakEval) | Automated evaluators for assessing jailbreak attempts | Python | MIT |
| [EasyJailbreak](https://github.com/thu-coai/EasyJailbreak) | Easy-to-use Python framework to generate adversarial jailbreak prompts | Python | Apache-2.0 |
| [Guardrails](https://github.com/ShreyaR/guardrails) | Add guardrails to large language models | Python | MIT |
| [LLM Guard](https://github.com/deadbits/llm-guard) | Security toolkit for LLM interactions | Python | Apache-2.0 |
| [AuditNLG](https://github.com/Alex-Fabbri/AuditNLG) | Reduce risks in generative AI systems for language | Python | MIT |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Toolkit for adding programmable guardrails to LLM conversational systems | Python | Apache-2.0 |
| [Garak](https://github.com/leondz/garak) | LLM vulnerability scanner | Python | MIT |
| [DeepTeam](https://github.com/DeepTeamAI/deepteam) | LLM red teaming framework | Python | Apache-2.0 |
| [MarkLLM](https://github.com/markllm/markllm) | Watermarking toolkit for LLM outputs | Python | Apache-2.0 |
| [LLMSanitize](https://github.com/llm-sanitize/llm-sanitize) | Security toolkit for sanitizing LLM inputs/outputs | Python | MIT |

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
| [LiteLLM](https://github.com/BerriAI/litellm) | Lightweight OpenAI-compatible gateway for multiple LLM providers | Python | MIT |
| [AI Gateway](https://github.com/Portkey-AI/ai-gateway) | Gateway for managing LLM requests, caching, and routing | Python | Apache-2.0 |
| [Langcorn](https://github.com/langcorn/langcorn) | Serve LangChain applications via FastAPI with production-ready endpoints | Python | MIT |
| [LitServe](https://github.com/Lightning-AI/LitServe) | High-speed GPU inference server with autoscaling and batch support | Python | Apache-2.0 |

### LLM Data Generation

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [DataDreamer](https://github.com/stanfordnlp/dreamer) | Framework for creating synthetic datasets to train & evaluate LLMs | Python | Apache-2.0 |
| [fabricator](https://github.com/fabricator-ai/fabricator) | Data generation toolkit for crafting synthetic training data | Python | MIT |
| [Promptwright](https://github.com/promptwright/promptwright) | Toolkit for prompt engineering, evaluation, and dataset curation | Python | Apache-2.0 |
| [EasyInstruct](https://github.com/zjunlp/EasyInstruct) | Instruction data generation framework for large-scale LLM training | Python | Apache-2.0 |
| [Text Machina](https://github.com/text-machina/text-machina) | Dataset generation framework for robust AI training | Python | Apache-2.0 |

### LLM Inference Platforms

| Platform | Description | Pricing | Features |
|----------|-------------|---------|----------|
| [Clarifai](https://www.clarifai.com/) | Lightning-fast compute for AI models & agents | Free tier + Pay-as-you-go | Pre-trained models, Deploy your own models on Dedicated compute, Model training, Workflow automation | 
| [Modal](https://modal.com/) | Serverless platform for AI/ML workloads | Pay-per-use | Serverless GPU, Auto-scaling |
| [Replicate](https://replicate.com/) | Run open-source models with a cloud API | Pay-per-use | Pre-built models, Custom training |
| [Together AI](https://www.together.ai/) | Cloud platform for open-source models | Various | Open models, Fine-tuning |
| [Anyscale](https://www.anyscale.com/) | Ray-based platform for AI applications | Enterprise | Distributed training, Serving |
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
