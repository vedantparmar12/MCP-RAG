#!/usr/bin/env python3
"""
Test script to verify Gemini API integration
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_gemini_setup():
    """Test if Gemini API is working correctly"""
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not set properly in .env file")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Test embedding generation
    try:
        print("🔍 Testing embedding generation...")
        result = genai.embed_content(
            model="models/text-embedding-004",
            content="Hello, this is a test"
        )
        embedding = result['embedding']
        print(f"✅ Embedding generated successfully! Dimension: {len(embedding)}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False
    
    # Test LLM generation
    try:
        print("🔍 Testing LLM generation...")
        model_choice = os.getenv("MODEL_CHOICE", "gemini-2.0-flash-exp")
        model = genai.GenerativeModel(model_choice)
        
        response = model.generate_content(
            "Hello! Please respond with a brief greeting.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=50
            )
        )
        print(f"✅ LLM generation successful! Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Gemini API integration is working correctly.")
    return True

if __name__ == "__main__":
    test_gemini_setup()
