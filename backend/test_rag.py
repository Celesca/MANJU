"""Test RAG functionality with AI Thailand Hackathon PDF"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Load environment
def load_env():
    # Load from backend/.env
    backend_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(backend_env):
        with open(backend_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # Load from root/.env (don't override existing)
    root_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(root_env):
        with open(root_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()

def test_rag_functionality():
    """Test RAG functionality with AI Thailand Hackathon questions"""
    
    load_env()
    
    print("Testing RAG functionality...")
    print(f"TOGETHER_API_KEY: {bool(os.getenv('TOGETHER_API_KEY'))}")
    print(f"PDF path: {os.path.join(os.path.dirname(__file__), 'documents', 'aithailand.pdf')}")
    
    # Check if PDF exists
    pdf_path = os.path.join(os.path.dirname(__file__), 'documents', 'aithailand.pdf')
    print(f"PDF exists: {os.path.exists(pdf_path)}")
    
    try:
        from MultiAgent import MultiAgent
        
        print("\n=== Initializing MultiAgent with RAG ===")
        ma = MultiAgent()
        print("✅ MultiAgent with RAG initialized successfully!")
        print(f"Provider: {ma.provider}")
        print(f"Model: {ma.config.model}")
        print(f"RAG tool available: {hasattr(ma, 'rag_tool')}")
        
        # Test questions about AI Thailand Hackathon
        test_questions = [
            "ทีมไหนชนะเลิศในการแข่งขัน AI Thailand Hackathon 2024 EP.2?",
            "รางวัลเงินสดรวมทั้งหมดเท่าไหร่?",
            "การแข่งขันจัดที่ไหนและเมื่อไหร่?",
            "มี API อะไรบ้างให้ใช้ในการแข่งขัน?",
            "สวัสดีครับ ผมสนใจข้อมูลทั่วไป"  # Non-hackathon question
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🧪 Test {i}: {question}")
            try:
                result = ma.run(question)
                print("✅ Success!")
                print(f"Response: {result['response']}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag_functionality()
