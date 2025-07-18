# MCP-RAG: An Intelligent RAG and Code Validation System

Welcome to MCP-RAG! This project is more than just a standard Retrieval-Augmented Generation (RAG) system. It's a powerful framework designed to give your AI agents the ability to learn from the web and to validate their own generated code against a trusted knowledge base, effectively preventing AI "hallucinations."

## Overview

At its core, the project has two main functions:

**Dynamic RAG Pipeline**: It can crawl websites, documentation, and even GitHub repositories, process the content, and store it in a vector database (Supabase). This knowledge is then made available to an AI agent through a set of simple tools, allowing it to answer questions with up-to-date, contextually relevant information.

**Knowledge Graph-based Code Validation**: In a more advanced use case, it can analyze an entire code repository and build a detailed "knowledge graph" in a Neo4j database. This graph maps out all the classes, functions, and their relationships. The system can then take AI-generated Python code and check it against this graph to ensure it's valid, catching errors where the AI might have "hallucinated" a function or used incorrect parameters.

This dual approach means you can not only feed your AI new information but also verify its output, creating a more reliable and trustworthy AI assistant.

## Key Features

- **Smart Web Crawler (Crawl4AI)**: Automatically detects the best way to crawl a URL, whether it's a standard webpage, a sitemap, or a text file.

- **Vector Database Integration**: Uses Supabase and pgvector to store and efficiently search through crawled content.

- **Hybrid Search**: Combines traditional keyword search with semantic vector search for more accurate and relevant retrieval results.

- **Agentic RAG for Code**: Can specifically identify and extract code blocks from documentation, creating a dedicated search index for code examples.

- **Hallucination Detection**: Leverages a Neo4j knowledge graph to validate Python code, ensuring correctness and preventing the use of non-existent APIs.

- **Flexible API Provider**: Easily switch between Google Gemini and OpenAI for generating embeddings and responses.

- **MCP Server**: Exposes its functionality as a set of tools for any Modular Agent-compliant client (like Claude).

## Setup and Installation

Follow these steps to get the project up and running.

### Step 1: Clone and Install Dependencies

First, get the code and install the required Python packages.

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
pip install -e .
```

The installation will include key libraries like mcp, crawl4ai, supabase, google-generativeai, and neo4j-driver.

### Step 2: Set Up Environment Variables

You'll need to configure your API keys and database credentials. Create a file named `.env` in the root of the project directory.

```bash
# .env

# --- API Provider ---
# Choose your provider: "gemini" or "openai"
API_PROVIDER="gemini"
# Your chosen model
MODEL_CHOICE="gemini-1.5-flash-latest"

# --- API Keys ---
# Add the key for your chosen provider
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"

# --- Supabase Database ---
# Get these from your Supabase project settings
SUPABASE_URL="your_supabase_url_here"
SUPABASE_SERVICE_KEY="your_supabase_service_key_here"

# --- Neo4j Knowledge Graph (Optional, for hallucination detection) ---
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_neo4j_password"

# --- RAG Strategy Booleans (true/false) ---
USE_CONTEXTUAL_EMBEDDINGS="false"
USE_HYBRID_SEARCH="true"
USE_RERANKING="true"
USE_AGENTIC_RAG="true"
USE_KNOWLEDGE_GRAPH="false"
```

### Step 3: Set Up the Supabase Database

You need to prepare your Supabase database by creating the necessary tables and functions for storing and searching content.

1. Go to your Supabase project's SQL Editor.
2. Open the `crawled_pages.sql` file from this repository.
3. Copy its entire content and run it in the SQL Editor.

This script will create the `crawled_pages`, `code_examples`, and `sources` tables, and it will set up the vector search functions, configured for the 768 dimensions used by Gemini embeddings.

### Step 4: Verify Your Setup

Run the provided test script to make sure everything is configured correctly. This will check your environment variables, database connection, and API key validity.

```bash
python test_mcp_functionality.py
```

If all tests pass, you're ready to go!

## How to Use the RAG Server

The primary way to interact with the project is by running the MCP server and connecting an AI agent to it.

### Step 1: Start the Server

Run the main server script. This will start listening for requests from your AI agent.

```bash
uv run src/crawl4ai_mcp.py
```

### Step 2: Connect Your MCP Client

Configure your agent (e.g., Claude Desktop, Windsurf) to connect to the running server. Add the following to your client's MCP configuration:

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

### Step 3: Use the Tools

Your agent now has access to a new set of tools. Here is a typical workflow:

**Crawl Content**: Start by feeding the system some knowledge. You can crawl a single page or an entire site.

```python
crawl_single_page(url='https://docs.python.org/3/library/asyncio.html')

smart_crawl_url(url='https://docs.python.org/3/tutorial/')
```

**Check Available Sources**: See what information has been indexed and is ready for querying.

```python
get_available_sources()
```

**Ask Questions (RAG)**: Perform a RAG query on the content you've crawled. You can search across all sources or filter by a specific one.

```python
perform_rag_query(query='How do I run an asyncio task?')

perform_rag_query(query='What are list comprehensions?', source='docs.python.org')
```

**Find Code Examples**: If you enabled `USE_AGENTIC_RAG`, you can specifically search for code snippets.

```python
search_code_examples(query='a simple web server in python')
```

## Advanced Feature: Hallucination Detection

For a more advanced use case, you can validate AI-generated code against a specific repository.

### Ingest a Repository

First, run the `parse_repo_into_neo4j.py` script to analyze a code repository and build the knowledge graph.

```bash
python knowledge_graphs/parse_repo_into_neo4j.py
```

### Detect Hallucinations

Now, run the `ai_hallucination_detector.py` script, pointing it to a Python file you want to validate (e.g., one written by an AI).

```bash
python knowledge_graphs/ai_hallucination_detector.py /path/to/your/ai_generated_script.py
```

The detector will analyze the script, cross-reference it with the knowledge graph, and generate a detailed report in both JSON and Markdown, highlighting any invalid API usage, non-existent methods, or incorrect parameters.
