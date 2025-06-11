#!/usr/bin/env python3
"""
AI Agent Toolkit - Main entry point
Multi-tool AI agent with web search, document retrieval, and SQL capabilities
"""

import logging
import sys
from src import AIAgent, Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the AI Agent"""
    
    print("🤖 AI Agent Toolkit")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("🔄 Initializing AI Agent...")
        agent = AIAgent()
        
        # Show available tools
        tools = agent.get_available_tools()
        print(f"✅ Agent ready! Available tools: {', '.join(tools)}")
        
        print("\n💡 Type your questions or 'quit' to exit")
        print("-" * 50)
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q", "bye"]:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Agent: ", end="", flush=True)
                response = agent.process_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"❌ Error: {e}")
                
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Make sure to set up your .env file with required API keys")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)