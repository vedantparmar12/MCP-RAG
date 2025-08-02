<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### Open-Source LLM Support (NEW)
- **No More GPT Dependency**: Replaced OpenAI GPT models with open-source alternatives
- **Ollama Integration**: Run powerful models like Mistral, Llama 2, and Gemma locally
- **HuggingFace Models**: Support for thousands of models including Phi-3, Mistral, and more
- **Quantization Support**: 4-bit and 8-bit quantization for running large models on consumer GPUs
- **Automatic Fallback**: Seamless switching between providers if one fails
- **Zero API Costs**: Run everything locally with Ollama for complete cost control

### Multi-Model Embedding Support (NEW)
- **OpenAI Embeddings**: Industry-standard embeddings with models like text-embedding-3-small/large
- **Cohere Embeddings**: Advanced embeddings with multilingual and multimodal support (v4.0)
- **Ollama Embeddings**: Fully local embeddings with no API costs using models like nomic-embed-text
- **HuggingFace Embeddings**: Flexible embeddings with both local and API options, supporting hundreds of models
- **Automatic Provider Selection**: Smart selection based on performance, cost, and availability
- **Fallback Support**: Automatic fallback to alternative providers if primary fails
- **Performance Tracking**: Monitors latency, success rates, and costs across providers

### Visual Document Processing with ColPali (NEW)
- **PDF & Image Understanding**: Process visual documents using state-of-the-art ColPali model
- **Late Interaction Search**: ColBERT-style retrieval for better accuracy on visual content
- **Hybrid RAG**: Combine visual and text search with intelligent reranking
- **Batch Processing**: Efficient parallel processing of multiple documents

### Core RAG Features
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content with configurable depth control
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously with memory-adaptive dispatching
- **Intelligent Content Chunking**: Smart markdown-aware chunking that respects code blocks, paragraphs, and sentences
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Management**: Automatic source summary generation and metadata tracking with word counts
- **Batch Processing**: Optimized batch insertion of documents with configurable batch sizes for performance
- **Concurrent Code Processing**: Parallel processing of code examples using ThreadPoolExecutor for faster indexing

### Self-Improvement Architecture (NEW)
- **Multi-Agent System**: Intelligent agents for dependency validation, code debugging, and integration testing
- **Evolution Orchestrator**: LangGraph-based workflow engine that coordinates the self-improvement process
- **Correctness Evaluation**: Comprehensive metrics including factual accuracy, ROUGE scores, nDCG, and code quality
- **Self-Healing Capabilities**: Automatic detection and resolution of system issues with rollback support
- **Feature Evolution**: System can analyze user requests, generate new features, test them, and deploy safely
- **Memory Persistence**: Uses Mem0 for tracking evolution history and learning from past improvements
- **Real-time System Monitoring**: CPU, memory, and disk usage tracking with performance metrics
- **API Cost Tracking**: Monitors and manages API usage costs for OpenAI and other services
- **Automated Test Generation**: Generates pytest-compatible tests for new features with coverage reporting
- **Security Validation**: AST-based code analysis to detect dangerous patterns before deployment

## Tools

The server provides essential web crawling and search tools:

### Core Tools (Always Available)

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

### Conditional Tools

5. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

### Self-Improvement Tools (NEW)

6. **`self_heal_system`**: Automatically diagnose and fix system issues using intelligent agents that validate dependencies, debug code, and test fixes before applying them.

7. **`evolve_rag_capability`**: Enable the system to self-improve by analyzing feature requests, crawling relevant documentation, generating implementation code, and safely deploying new features after thorough testing.

8. **`perform_rag_query_with_metrics`**: Enhanced RAG query that includes correctness evaluation metrics (factual accuracy, ROUGE scores, nDCG ranking quality, and code quality assessment).

9. **`get_system_metrics`**: Get comprehensive system metrics, evaluation history, and performance insights to monitor and improve the RAG system over time.

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   # Basic installation (OpenAI embeddings only)
   uv pip install -e .
   
   # With Cohere support
   uv pip install -e ".[cohere]"
   
   # With HuggingFace support
   uv pip install -e ".[huggingface]"
   
   # With all providers (embeddings + LLMs + visual)
   uv pip install -e ".[all]"
   
   # With visual document support (ColPali)
   uv pip install -e ".[colpali]"
   
   # With quantization support for large models
   uv pip install -e ".[quantization]"
   
   # Run Crawl4AI setup
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# Embedding Provider Configuration
EMBEDDING_PROVIDER=auto  # Options: openai, cohere, ollama, huggingface, auto

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002

# Cohere Configuration (optional)
COHERE_API_KEY=your_cohere_api_key
COHERE_EMBEDDING_MODEL=embed-english-v3.0  # Options: embed-english-v3.0, embed-multilingual-v3.0, embed-v4.0
COHERE_USE_V2=false  # Set to true for multimodal support with embed-v4.0

# Ollama Configuration (optional - for local embeddings)
ENABLE_OLLAMA=false  # Set to true to enable Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # Options: nomic-embed-text, all-minilm, mxbai-embed-large

# HuggingFace Configuration (optional)
ENABLE_HUGGINGFACE=false  # Set to true to enable HuggingFace
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_USE_API=false  # Set to true to use HF Inference API instead of local
HUGGINGFACE_API_TOKEN=your_hf_api_token  # Required only if USE_API=true
HUGGINGFACE_DEVICE=cpu  # Options: cpu, cuda, mps

# LLM Provider Configuration (NEW - Open Source Models)
LLM_PROVIDER=auto  # Options: huggingface, ollama, auto

# Ollama LLM Configuration (recommended for local inference)
ENABLE_OLLAMA_LLM=true
OLLAMA_LLM_MODEL=mistral:instruct  # Options: mistral:instruct, llama2:7b-chat, phi, gemma:7b-instruct

# HuggingFace LLM Configuration
ENABLE_HUGGINGFACE_LLM=true
HUGGINGFACE_LLM_MODEL=microsoft/Phi-3-mini-4k-instruct  # Small but capable model
HUGGINGFACE_LLM_USE_API=false  # Set to true to use HF Inference API
HUGGINGFACE_LLM_DEVICE=cpu  # Options: cpu, cuda, mps
HUGGINGFACE_LLM_4BIT=false  # Enable 4-bit quantization (requires GPU)
HUGGINGFACE_LLM_8BIT=false  # Enable 8-bit quantization (requires GPU)

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `MODEL_CHOICE`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries using parallel processing, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: 
  - Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations
  - Parallel processing of code summaries using ThreadPoolExecutor for faster indexing
  - Context-aware summaries that include surrounding documentation text

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

### Embedding Provider Configuration

The Crawl4AI RAG MCP server now supports multiple embedding providers, each with different strengths:

#### **OpenAI** (Default)
- **Best for**: High-quality embeddings with good all-around performance
- **Models**: text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims)
- **Cost**: $0.020 - $0.130 per 1M tokens
- **Setup**: Just set `OPENAI_API_KEY`

#### **Cohere**
- **Best for**: Multilingual content and multimodal embeddings (text + images)
- **Models**: embed-english-v3.0, embed-multilingual-v3.0, embed-v4.0 (multimodal)
- **Cost**: ~$0.100 per 1M tokens
- **Setup**: Set `COHERE_API_KEY` and `ENABLE_COHERE=true`
- **Note**: Use `COHERE_USE_V2=true` for multimodal support

#### **Ollama** (Local)
- **Best for**: Privacy-conscious users, no API costs, offline operation
- **Models**: nomic-embed-text (768 dims), mxbai-embed-large (1024 dims), etc.
- **Cost**: FREE (runs locally)
- **Setup**: Install Ollama, run `ollama pull nomic-embed-text`, set `ENABLE_OLLAMA=true`

#### **HuggingFace**
- **Best for**: Flexibility, hundreds of model options, both local and API
- **Models**: all-MiniLM-L6-v2 (384 dims), all-mpnet-base-v2 (768 dims), etc.
- **Cost**: FREE for local, minimal for API
- **Setup**: Set `ENABLE_HUGGINGFACE=true`, optionally install sentence-transformers

### LLM Provider Configuration

The Crawl4AI RAG MCP server now uses open-source LLMs instead of OpenAI GPT models:

#### **Ollama** (Recommended for most users)
- **Best for**: Local inference, privacy, zero API costs
- **Models**: Mistral 7B Instruct, Llama 2 7B, Phi-2, Gemma 7B
- **Setup**: Install Ollama, run `ollama pull mistral:instruct`
- **Performance**: Fast on modern CPUs, faster with GPU

#### **HuggingFace** 
- **Best for**: Access to cutting-edge models, quantization options
- **Models**: Phi-3, Mistral, Llama 2, Mixtral, and thousands more
- **Setup**: Automatic model download on first use
- **Options**: 4-bit/8-bit quantization for large models on consumer GPUs

### Recommended Configurations

**For general documentation RAG:**
```
EMBEDDING_PROVIDER=openai
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
EMBEDDING_PROVIDER=openai
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For fast, local, privacy-focused RAG:**
```
# Embeddings
EMBEDDING_PROVIDER=ollama
ENABLE_OLLAMA=true
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# LLMs
LLM_PROVIDER=ollama
ENABLE_OLLAMA_LLM=true
OLLAMA_LLM_MODEL=mistral:instruct

# RAG Settings
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

**For powerful local setup with GPU:**
```
# Embeddings
EMBEDDING_PROVIDER=huggingface
ENABLE_HUGGINGFACE=true
HUGGINGFACE_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# LLMs
LLM_PROVIDER=huggingface
ENABLE_HUGGINGFACE_LLM=true
HUGGINGFACE_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HUGGINGFACE_LLM_DEVICE=cuda
HUGGINGFACE_LLM_4BIT=true  # Use 4-bit quantization

# RAG Settings
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For multilingual content:**
```
EMBEDDING_PROVIDER=cohere
COHERE_EMBEDDING_MODEL=embed-multilingual-v3.0
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## Self-Improvement Architecture Details

### Core Components

#### 1. **Version Control & Rollback System**
- Git-based version tracking for all system changes
- Automatic checkpoint creation before deployments
- Rollback capabilities for failed evolutions
- Branch management for feature development
- Evolution history tracking with unique IDs

#### 2. **Security Sandbox**
- Code validation with AST analysis
- Detection of dangerous patterns (exec, eval, subprocess, file operations)
- Sandboxed execution environment with Docker integration
- Security risk assessment for generated code
- Configurable security rules and thresholds

#### 3. **Resource Management**
- Rate limiting for API calls and evolution requests
- Cost tracking for API usage (OpenAI, Supabase) with configurable models
- System resource monitoring (CPU, memory, disk) using psutil
- User quota management with hourly/daily limits
- Circuit breakers for external service failures

#### 4. **Automated Testing Framework**
- Automatic test generation for new features using LLM
- Integration with pytest for test execution
- Coverage reporting and threshold enforcement
- Regression test suite management
- Performance benchmarking for new features

#### 5. **Agent Orchestration**
- LangGraph-based workflow management with conditional edges
- Coordinated agent execution with state management
- Error recovery and retry mechanisms with exponential backoff
- Memory persistence with Mem0
- Parallel agent execution where applicable

### Usage Examples

#### Self-Healing System Issue
```python
# Example: Fix a dependency issue
result = await self_heal_system(
    issue_description="ImportError: langgraph module not found"
)
```

#### Evolving New Capability
```python
# Example: Add a new RAG feature
result = await evolve_rag_capability(
    feature_request="Add support for PDF document crawling and indexing",
    documentation_urls=["https://pypdf2.readthedocs.io/"]
)
```

#### Running Query with Metrics
```python
# Example: Perform RAG query with evaluation
result = await perform_rag_query_with_metrics(
    query="How to implement OAuth in FastAPI?",
    source="fastapi.tiangolo.com",
    enable_evaluation=True
)
```

## Additional Implementation Details

### Performance Optimizations
- **Parallel Content Processing**: Uses ThreadPoolExecutor for concurrent processing of embeddings and code summaries
- **Batch Operations**: Configurable batch sizes (default 20) for efficient database insertions
- **Memory-Adaptive Crawling**: MemoryAdaptiveDispatcher monitors system resources during crawling
- **Smart Chunking**: Respects markdown structure, code blocks, and natural text boundaries

### Agent Implementation Details
- **BaseAgent Class**: Abstract base class for all agents with state management and validation
- **Dependency Validator**: Checks Python dependencies, validates versions, and auto-fixes issues
- **Code Debugger**: AST-based syntax validation, automatic error fixing, and code quality checks
- **Integration Tester**: Runs unit tests, integration tests, and monitors performance metrics

### Database Schema Enhancements
- **Sources Table**: Tracks unique sources with summaries and word counts
- **Code Examples Table**: Dedicated table for code snippets with summaries and metadata
- **Improved Indexing**: Optimized indexes for hybrid search performance

### Error Handling & Recovery
- **Circuit Breakers**: Automatic failure detection and service isolation
- **Exponential Backoff**: Smart retry logic for transient failures
- **Graceful Degradation**: System continues with reduced functionality when services fail
- **Comprehensive Logging**: Detailed error tracking and diagnostics

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers
5. Implement custom agents by extending the BaseAgent class
6. Add new RAG strategies in the evaluation system
7. Customize the security sandbox for your specific requirements
8. Extend the resource management system with custom limits