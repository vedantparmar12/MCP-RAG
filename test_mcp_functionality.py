#!/usr/bin/env python3
"""
Comprehensive test script for MCP Crawl4AI RAG functionality
"""
import os
import sys
import json
import time
import asyncio
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test if all required environment variables are set"""
    print("🔍 Testing environment setup...")
    
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY',
        'API_PROVIDER',
    ]
    
    api_provider = os.getenv('API_PROVIDER', 'openai')
    if api_provider == 'openai':
        required_vars.append('OPENAI_API_KEY')
    elif api_provider == 'gemini':
        required_vars.append('GEMINI_API_KEY')
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_supabase_connection():
    """Test Supabase connection"""
    print("🔍 Testing Supabase connection...")
    
    try:
        from supabase import create_client
        
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_KEY')
        
        supabase = create_client(url, key)
        
        # Test connection by querying a simple table
        result = supabase.table('crawled_pages').select('id').limit(1).execute()
        print("✅ Supabase connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_crawl4ai_import():
    """Test if Crawl4AI can be imported and initialized"""
    print("🔍 Testing Crawl4AI import...")
    
    try:
        from crawl4ai import AsyncWebCrawler
        print("✅ Crawl4AI imported successfully")
        return True
    except Exception as e:
        print(f"❌ Crawl4AI import failed: {e}")
        return False

async def test_basic_crawling():
    """Test basic crawling functionality"""
    print("🔍 Testing basic crawling...")
    
    try:
        from crawl4ai import AsyncWebCrawler
        
        async with AsyncWebCrawler() as crawler:
            # Test with a simple, reliable webpage
            result = await crawler.arun(
                url="https://httpbin.org/html",
                bypass_cache=True
            )
            
            if result.success and result.markdown:
                print("✅ Basic crawling successful")
                print(f"   Crawled content length: {len(result.markdown)} characters")
                return True
            else:
                print("❌ Basic crawling failed - no content returned")
                return False
                
    except Exception as e:
        print(f"❌ Basic crawling failed: {e}")
        return False

def test_mcp_server_import():
    """Test MCP server import"""
    print("🔍 Testing MCP server import...")
    
    try:
        import crawl4ai_mcp
        print("✅ MCP server module imported successfully")
        
        # Check if main functions exist
        if hasattr(crawl4ai_mcp, 'mcp'):
            print("✅ MCP instance found")
        else:
            print("❌ MCP instance not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ MCP server import failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation"""
    print("🔍 Testing embedding generation...")
    
    try:
        api_provider = os.getenv('API_PROVIDER', 'openai')
        
        if api_provider == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="Test embedding generation"
            )
            
            embedding = response.data[0].embedding
            print(f"✅ OpenAI embedding generated successfully! Dimension: {len(embedding)}")
            
        elif api_provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content="Test embedding generation"
            )
            
            embedding = result['embedding']
            print(f"✅ Gemini embedding generated successfully! Dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_database_tables():
    """Test if required database tables exist"""
    print("🔍 Testing database tables...")
    
    try:
        from supabase import create_client
        
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_KEY')
        
        supabase = create_client(url, key)
        
        # Test main table
        result = supabase.table('crawled_pages').select('*').limit(1).execute()
        print("✅ crawled_pages table exists")
        
        # Test code examples table if agentic RAG is enabled
        if os.getenv('USE_AGENTIC_RAG', 'false').lower() == 'true':
            result = supabase.table('code_examples').select('*').limit(1).execute()
            print("✅ code_examples table exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Database tables test failed: {e}")
        print("   Make sure you've run the crawled_pages.sql script in your Supabase database")
        return False

def create_test_summary():
    """Create a summary of test results"""
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Supabase Connection", test_supabase_connection),
        ("Database Tables", test_database_tables),
        ("Crawl4AI Import", test_crawl4ai_import),
        ("MCP Server Import", test_mcp_server_import),
        ("Embedding Generation", test_embedding_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n📋 Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your MCP Crawl4AI RAG project is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

async def test_basic_crawling_async():
    """Async wrapper for basic crawling test"""
    return await test_basic_crawling()

def main():
    """Main test function"""
    print("🚀 Starting MCP Crawl4AI RAG Project Tests")
    print("="*50)
    
    # Run basic tests
    success = create_test_summary()
    
    # Run async crawling test
    print("\n🔍 Testing basic crawling (async)...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        crawl_result = loop.run_until_complete(test_basic_crawling_async())
        loop.close()
        
        if crawl_result:
            print("✅ Async crawling test passed")
        else:
            print("❌ Async crawling test failed")
            success = False
    except Exception as e:
        print(f"❌ Async crawling test failed: {e}")
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED! Your project is working correctly.")
        print("\n📝 Next steps:")
        print("1. Start the MCP server: uv run src/crawl4ai_mcp.py")
        print("2. Connect to it from your MCP client (Claude, Windsurf, etc.)")
        print("3. Try crawling a webpage with the crawl_single_page tool")
        print("4. Test RAG queries with the perform_rag_query tool")
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")
    
    return success

if __name__ == "__main__":
    main()
