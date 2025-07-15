#!/usr/bin/env python3
"""
Simple test examples for MCP Crawl4AI RAG functionality
This script provides examples of how to test the MCP server once it's running.
"""

import json

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"[{title}]")
    print(f"{'='*60}")

def print_example(title, description, content):
    """Print a formatted example"""
    print(f"\n* {title}")
    print(f"  {description}")
    print(f"  Content: {content}")

def main():
    """Main function to display test examples"""
    print("MCP Crawl4AI RAG Server - Test Examples")
    print("=" * 60)
    
    print_section("1. MCP Server Configuration Examples")
    
    print("\nFor Claude Desktop/Windsurf (SSE):")
    sse_config = {
        "mcpServers": {
            "crawl4ai-rag": {
                "transport": "sse",
                "url": "http://localhost:8051/sse"
            }
        }
    }
    print(json.dumps(sse_config, indent=2))
    
    print("\nFor Windsurf (use serverUrl instead):")
    windsurf_config = {
        "mcpServers": {
            "crawl4ai-rag": {
                "transport": "sse", 
                "serverUrl": "http://localhost:8051/sse"
            }
        }
    }
    print(json.dumps(windsurf_config, indent=2))
    
    print("\nFor Claude Code CLI:")
    print("claude mcp add-json crawl4ai-rag '{\"type\":\"http\",\"url\":\"http://localhost:8051/sse\"}' --scope user")
    
    print_section("2. Testing Commands (Run these after starting the server)")
    
    print("\nStart the MCP server:")
    print("uv run src/crawl4ai_mcp.py")
    print("(Keep this running in a separate terminal)")
    
    print_section("3. Test URLs for Crawling")
    
    test_urls = [
        ("Simple HTML page", "https://httpbin.org/html", "Basic HTML structure test"),
        ("Documentation site", "https://docs.python.org/3/tutorial/", "Real documentation crawling"),
        ("GitHub README", "https://raw.githubusercontent.com/microsoft/vscode/main/README.md", "Markdown content"),
        ("API documentation", "https://jsonplaceholder.typicode.com/", "API docs example"),
        ("Blog post", "https://example.com", "Simple example site"),
    ]
    
    for name, url, description in test_urls:
        print_example(name, description, url)
    
    print_section("4. MCP Tool Examples")
    
    print("\nAvailable MCP Tools:")
    tools = [
        ("crawl_single_page", "Crawl a single webpage", "crawl_single_page(url='https://httpbin.org/html')"),
        ("smart_crawl_url", "Smart crawl with auto-detection", "smart_crawl_url(url='https://docs.python.org/3/')"),
        ("get_available_sources", "List crawled sources", "get_available_sources()"),
        ("perform_rag_query", "Search crawled content", "perform_rag_query(query='python functions', limit=5)"),
    ]
    
    for tool_name, description, example in tools:
        print_example(tool_name, description, example)
    
    print_section("5. Testing RAG Queries")
    
    rag_examples = [
        ("Basic search", "Search for general concepts", "How to create a Python function"),
        ("Technical search", "Search for specific terms", "async await python"),
        ("Code examples", "Find code snippets", "python class example"),
        ("Documentation", "Find documentation", "python tutorial basics"),
        ("Error handling", "Find error examples", "python exception handling"),
    ]
    
    for query_type, description, example in rag_examples:
        print_example(query_type, description, f"perform_rag_query(query='{example}', limit=3)")
    
    print_section("6. Advanced Features Testing")
    
    print("\nTest with different RAG strategies:")
    print("   Modify your .env file to test different configurations:")
    print("   - USE_HYBRID_SEARCH=true (recommended)")
    print("   - USE_RERANKING=true (recommended)")
    print("   - USE_CONTEXTUAL_EMBEDDINGS=true (slower but more accurate)")
    print("   - USE_AGENTIC_RAG=true (for code examples)")
    print("   - USE_KNOWLEDGE_GRAPH=true (for hallucination detection)")
    
    print_section("7. Sample Test Workflow")
    
    workflow = [
        "1. Start server: uv run src/crawl4ai_mcp.py",
        "2. Connect from your MCP client (Claude, Windsurf, etc.)",
        "3. Test crawling: crawl_single_page('https://httpbin.org/html')",
        "4. Check sources: get_available_sources()",
        "5. Test RAG: perform_rag_query('test content', limit=3)",
        "6. Try smart crawling: smart_crawl_url('https://docs.python.org/3/')",
        "7. Advanced search: perform_rag_query('python functions', source='docs.python.org')",
    ]
    
    for step in workflow:
        print(f"   {step}")
    
    print_section("8. Troubleshooting Commands")
    
    print("\nIf server won't start:")
    print("   - Check port: netstat -an | findstr :8051")
    print("   - Test environment: python test_mcp_functionality.py")
    print("   - Check logs in terminal where server is running")
    
    print("\nIf crawling fails:")
    print("   - Test simple URL first: https://httpbin.org/html")
    print("   - Check internet connection")
    print("   - Some sites may block crawling")
    
    print("\nIf RAG queries return no results:")
    print("   - Make sure you've crawled some content first")
    print("   - Check get_available_sources() output")
    print("   - Try broader search terms")
    
    print_section("9. Performance Testing")
    
    print("\nPerformance test commands:")
    print("   - Small site: crawl_single_page('https://httpbin.org/html')")
    print("   - Medium site: smart_crawl_url('https://docs.python.org/3/tutorial/')")
    print("   - Large site: smart_crawl_url('https://docs.python.org/3/') (full docs)")
    print("   - Batch queries: Multiple perform_rag_query calls")
    
    print_section("10. Integration Testing")
    
    print("\nTest with different MCP clients:")
    print("   - Claude Desktop: Add to settings.json")
    print("   - Windsurf: Add to MCP settings")
    print("   - Claude Code CLI: Use the command above")
    print("   - Custom MCP client: Use SSE endpoint")
    
    print(f"\n{'='*60}")
    print("Ready to test! Start the server and try these examples.")
    print(f"{'='*60}")
    
    # Also show a quick test command
    print("\nQuick Test Command:")
    print("python -c \"import sys; sys.path.append('src'); import crawl4ai_mcp; print('MCP server ready!')\"")

if __name__ == "__main__":
    main()
