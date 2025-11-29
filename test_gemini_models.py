"""
Quick diagnostic to check which Gemini models are available with your API key
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    exit(1)

print(f"✅ API Key found: {api_key[:20]}...")
print("\n📋 Checking available models...\n")

try:
    genai.configure(api_key=api_key)
    
    # List all available models
    models = genai.list_models()
    
    vision_models = []
    text_models = []
    
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            if 'vision' in model.name.lower() or 'pro' in model.name.lower() or 'flash' in model.name.lower():
                vision_models.append(model.name)
            else:
                text_models.append(model.name)
    
    print("🖼️  VISION/MULTIMODAL MODELS (for receipt analysis):")
    if vision_models:
        for model in vision_models:
            print(f"  ✓ {model}")
    else:
        print("  ❌ No vision models found!")
    
    print("\n📝 TEXT-ONLY MODELS:")
    if text_models:
        for model in text_models:
            print(f"  ✓ {model}")
    else:
        print("  ❌ No text models found!")
    
    print("\n" + "="*60)
    
    if vision_models:
        recommended = vision_models[0]
        print(f"\n✅ RECOMMENDED MODEL TO USE: {recommended}")
        print(f"\nUpdate vision_agent.py line 22 to:")
        print(f'   self.model = genai.GenerativeModel("{recommended}")')
    else:
        print("\n❌ YOUR API KEY HAS NO MODEL ACCESS!")
        print("   Please regenerate your Gemini API key at:")
        print("   https://aistudio.google.com/app/apikey")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("\nYour API key might be invalid or expired.")
    print("Generate a new one at: https://aistudio.google.com/app/apikey")
