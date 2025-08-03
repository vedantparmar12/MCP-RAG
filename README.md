# MCP-RAG: Model Context Protocol with Retrieval Augmented Generation

## Overview

MCP-RAG is an advanced Python-based system that combines the Model Context Protocol (MCP) with Retrieval Augmented Generation (RAG) capabilities. This project provides a comprehensive framework for intelligent web crawling, content processing, multi-modal data handling, and automated code evolution with integrated testing and evaluation systems.

## 🚀 Key Features

- **Intelligent Web Crawling**: Advanced crawling capabilities using Crawl4AI
- **Multi-Modal Processing**: Support for text, visual, and structured data processing
- **Multiple LLM Providers**: Integration with OpenAI, Ollama, and HuggingFace models
- **Multiple Embedding Providers**: Support for OpenAI, Cohere, HuggingFace, and Ollama embeddings
- **Automated Code Evolution**: Self-improving codebase with version control
- **Comprehensive Testing**: Automated testing and integration validation
- **Security Sandbox**: Secure code execution environment
- **Performance Evaluation**: Built-in correctness evaluation systems
- **Resource Management**: Intelligent resource allocation and monitoring

## 📁 Project Structure

```
MCP-RAG/
├── src/
│   ├── crawl4ai_mcp.py          # Main MCP server implementation (1,450 lines)
│   ├── utils.py                 # Utility functions and helpers
│   ├── agents/                  # AI agent implementations
│   │   ├── base_agent.py        # Base agent class
│   │   ├── code_debugger.py     # Code debugging agent
│   │   ├── dependency_validator.py  # Dependency validation agent
│   │   ├── evolution_orchestrator.py  # Code evolution orchestrator
│   │   └── integration_tester.py    # Integration testing agent
│   ├── embeddings/              # Embedding provider implementations
│   │   ├── base.py              # Base embedding interface
│   │   ├── cohere_provider.py   # Cohere embeddings
│   │   ├── huggingface_provider.py  # HuggingFace embeddings
│   │   ├── manager.py           # Embedding provider manager
│   │   ├── ollama_provider.py   # Ollama embeddings
│   │   └── openai_provider.py   # OpenAI embeddings
│   ├── evaluation/              # Evaluation and metrics
│   │   └── correctness_evaluator.py  # Code correctness evaluation
│   ├── evolution/               # Code evolution system
│   │   └── version_control.py   # Version control management
│   ├── llm/                     # Large Language Model providers
│   │   ├── base.py              # Base LLM interface
│   │   ├── huggingface_llm.py   # HuggingFace LLM integration
│   │   ├── manager.py           # LLM provider manager
│   │   └── ollama_llm.py        # Ollama LLM integration
│   ├── resource_management/     # Resource allocation and monitoring
│   │   └── manager.py           # Resource manager
│   ├── security/                # Security and sandboxing
│   │   └── sandbox.py           # Secure execution environment
│   ├── testing/                 # Automated testing framework
│   │   └── automated_tester.py  # Automated test execution
│   └── visual/                  # Visual content processing
│       └── colpali_processor.py # ColPaLi visual document processing
├── crawled_pages.sql            # Database schema for crawled pages
├── crawled_pages_multimodel.sql # Multi-model database schema
└── Implementation.md            # Detailed implementation guide
```

## 🏗️ Architecture Overview

### Core Components

**1. MCP Server (`crawl4ai_mcp.py`)**
- Main entry point and MCP protocol implementation
- Handles client connections and request routing
- Coordinates between different system components
- Provides unified API for all functionality

**2. Agent System (`agents/`)**
- **Base Agent**: Foundation class for all AI agents
- **Code Debugger**: Automated code analysis and debugging
- **Dependency Validator**: Ensures code dependencies are correct
- **Evolution Orchestrator**: Manages automated code improvements
- **Integration Tester**: Validates component integration

**3. Multi-Provider Architecture**
- **LLM Providers**: Supports OpenAI, Ollama, and HuggingFace models
- **Embedding Providers**: Multiple embedding backends for flexibility
- **Unified Management**: Centralized provider management and switching

**4. Security & Resource Management**
- **Sandbox Environment**: Secure code execution with resource limits
- **Resource Monitor**: Tracks CPU, memory, and GPU usage
- **Access Control**: Manages file system and network access

### Data Flow

1. **Input Processing**: Web content crawled and processed
2. **Multi-Modal Analysis**: Text, images, and structured data extracted
3. **Embedding Generation**: Content converted to vector representations
4. **LLM Processing**: Intelligent analysis and response generation
5. **Quality Assurance**: Automated testing and evaluation
6. **Evolution**: Continuous improvement based on feedback

## 🔧 Key Modules

### Web Crawling & Content Processing
```python
# Main crawling functionality with advanced features
- Intelligent content extraction
- Multi-modal content handling
- Rate limiting and respect for robots.txt
- Content deduplication
- Structured data extraction
```

### Embedding System
```python
# Multiple provider support with unified interface
- OpenAI embeddings (text-embedding-ada-002, text-embedding-3-small/large)
- Cohere embeddings (embed-english-v3.0, embed-multilingual-v3.0)
- HuggingFace embeddings (sentence-transformers models)
- Ollama local embeddings (llama2, mistral, etc.)
```

### LLM Integration
```python
# Flexible LLM provider management
- OpenAI models (GPT-3.5, GPT-4, etc.)
- Ollama local models (Llama2, Mistral, Code Llama)
- HuggingFace models (various open-source models)
- Automatic fallback and load balancing
```

### Visual Processing
```python
# Advanced visual document understanding
- ColPaLi integration for visual document processing
- Image text extraction and analysis
- Document layout understanding
- Multi-modal embedding generation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- SQLite (for data storage)
- Optional: GPU support for local model inference

### Dependencies
The project uses a modular approach with optional dependencies based on providers:

**Core Dependencies:**
- `crawl4ai`: Web crawling framework
- `sqlite3`: Database operations
- `asyncio`: Asynchronous operations
- `json`: Data serialization

**Optional Provider Dependencies:**
- `openai`: OpenAI API integration
- `cohere`: Cohere API integration
- `transformers`: HuggingFace models
- `torch`: PyTorch for local inference
- `ollama`: Local LLM serving

### Configuration
1. Set up API keys for external providers (OpenAI, Cohere)
2. Configure database connection strings
3. Set resource limits and security policies
4. Initialize embedding and LLM providers

## 📊 Performance Metrics

**Codebase Statistics:**
- **Total Files**: 41 files
- **Total Lines of Code**: 9,602 lines
- **Primary Language**: Python (32 files)
- **Database Scripts**: 2 SQL files
- **Documentation**: Implementation guide included

**Component Distribution:**
- **Core MCP Server**: 1,450 lines (largest component)
- **Utilities**: 850+ lines of helper functions
- **Agent System**: 6 specialized agents
- **Provider Integrations**: 15+ provider implementations
- **Testing & Evaluation**: Comprehensive test suites

## 🔍 Usage Examples

### Basic Web Crawling
```python
# Initialize MCP server with crawling capabilities
server = MCPServer()
result = await server.crawl_url("https://example.com")
```

### Multi-Modal Content Processing
```python
# Process documents with visual elements
processor = ColPaLiProcessor()
embeddings = await processor.process_document(document_path)
```

### Automated Code Evolution
```python
# Enable self-improving capabilities
orchestrator = EvolutionOrchestrator()
await orchestrator.evolve_codebase()
```

## 🧪 Testing & Quality Assurance

The project includes comprehensive testing frameworks:

- **Automated Testing**: Continuous integration testing
- **Correctness Evaluation**: Code quality metrics
- **Integration Testing**: Cross-component validation
- **Performance Testing**: Resource usage monitoring
- **Security Testing**: Sandbox validation

## 🔒 Security Features

- **Sandboxed Execution**: Isolated code execution environment
- **Resource Limits**: CPU, memory, and time constraints
- **Access Control**: Restricted file system and network access
- **Input Validation**: Comprehensive input sanitization
- **Audit Logging**: Complete operation tracking

## 🔄 Evolution & Maintenance

The project includes automated evolution capabilities:

- **Version Control Integration**: Automated git operations
- **Dependency Management**: Smart dependency updates
- **Code Quality Improvement**: Automated refactoring
- **Performance Optimization**: Continuous performance tuning
- **Documentation Updates**: Automated documentation generation

## 📈 Future Enhancements

- **Multi-Agent Collaboration**: Enhanced agent coordination
- **Real-time Processing**: Streaming data processing
- **Advanced Visual Understanding**: Improved multi-modal capabilities
- **Distributed Processing**: Scalable architecture
- **Custom Model Training**: Domain-specific model fine-tuning

## 🤝 Contributing

This project follows a modular architecture that makes it easy to extend:

1. **Adding New Providers**: Implement the base interface for LLM or embedding providers
2. **Creating New Agents**: Extend the BaseAgent class for specialized functionality
3. **Enhancing Security**: Add new sandbox policies and security measures
4. **Improving Performance**: Optimize resource management and processing pipelines

## 📝 License & Documentation

For detailed implementation information, refer to the `Implementation.md` file included in the repository. The project maintains comprehensive documentation for all major components and provides examples for common use cases.

---

*This documentation was automatically generated from codebase analysis and provides a comprehensive overview of the MCP-RAG system architecture and capabilities.*
