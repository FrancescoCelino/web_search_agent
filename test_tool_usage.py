"""
Test script to verify that tools are actually being used
"""

import logging
import sys
from src import AIAgent, Config

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,  # Cambiato da DEBUG a INFO per output più pulito
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def simple_test():
    """Test semplice per verificare i tool"""
    print("\n🔬 SIMPLE TOOL TEST")
    print("="*50)
    
    try:
        agent = AIAgent()
        print(f"✅ Agent initialized with {len(agent.tools)} tools")
        
        # Test 1: Query che dovrebbe usare web search
        print("\n🧪 Test 1: Web Search")
        print("-" * 30)
        query1 = "search the internet for current news about artificial intelligence"
        print(f"Query: {query1}")
        response1 = agent.process_query(query1)
        print(f"Response length: {len(response1)} characters")
        print(f"Response: {response1}")
        
        # Test 2: Query semplice che non dovrebbe usare tool
        print("\n🧪 Test 2: Simple Math (no tools)")
        print("-" * 30)
        query2 = "what is 5 + 3?"
        print(f"Query: {query2}")
        response2 = agent.process_query(query2)
        print(f"Response: {response2}")
        
        # Test 3: Query SQL
        print("\n🧪 Test 3: Database Query")
        print("-" * 30)
        query3 = "how many customers are in the chinook database?"
        print(f"Query: {query3}")
        response3 = agent.process_query(query3)
        print(f"Response length: {len(response3)} characters")
        print(f"Response: {response3}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def debug_tools():
    """Debug dei tool disponibili"""
    print("\n🔧 TOOL DEBUG")
    print("="*50)
    
    try:
        from src.tools import ToolManager
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model=Config.MODEL_NAME)
        manager = ToolManager(llm)
        tools = manager.get_all_tools()
        
        print(f"🔧 Total tools: {len(tools)}")
        for i, tool in enumerate(tools):
            tool_name = getattr(tool, 'name', 'Unknown')
            tool_desc = getattr(tool, 'description', 'No description')[:50]
            print(f"  {i+1}. {tool_name}: {tool_desc}...")
            
    except Exception as e:
        print(f"❌ Tool debug failed: {e}")

if __name__ == "__main__":
    print("🚀 STARTING TOOL VERIFICATION")
    print("Look for these indicators:")
    print("  🌐 WEB SEARCH TOOL CALLED")
    print("  🗄️ SQL TOOL CALLED") 
    print("  📚 DOCUMENT RETRIEVAL TOOL CALLED")
    print("  🔧 TOOLS NODE EXECUTED")
    print("  💬 Direct response (no tools)")
    
    debug_tools()
    simple_test()
    
    print("\n✅ Tests completed!")
    print("Check the logs above to see which tools were actually used.")