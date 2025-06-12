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
    level=logging.DEBUG, # level=logging.INFO for less verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the AI Agent"""
    
    print("AI Agent Toolkit")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("Initializing AI Agent...")
        agent = AIAgent()
        
        # Show available tools
        tools = agent.get_available_tools()
        print(f"Agent ready! Available tools: {', '.join(tools)}")
        
        print("\nCommands:")
        print("  - Type your questions")
        print("  - 'quit' or 'exit' to leave")
        print("  - 'debug on/off' to toggle detailed responses")
        print("  - 'history' to see conversation history")
        print("  - 'clear' to clear conversation memory")
        print("  - 'summary' to see conversation summary")
        print("-" * 50)
        
        debug_mode = False
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q", "bye"]:
                    print("Goodbye!")
                    break
                
                # Toggle debug mode
                if user_input.lower() == "debug on":
                    debug_mode = True
                    print("Debug mode ON - will show full responses")
                    continue
                elif user_input.lower() == "debug off":
                    debug_mode = False
                    print("Debug mode OFF - normal responses")
                    continue

                # memory commands
                elif user_input.lower() == "history":
                    history = agent.get_conversation_history
                    if not history:
                        print("No history available.")
                    else:
                        print(f"Convo history ({len(history)} messages):")
                        print("-" * 60)
                        for role, content in history: 
                            prefix = "👤" if role == "user" else "🤖"
                            print(f"{prefix} {role}: {content[:150]}{'...' if len(content) > 150 else ''}")
                        print("-" * 60)
                    continue

                elif user_input.lower() == "clear" or user_input.lower() == "clr":
                    agent.clear_conversation()
                    print("Memory cleared")
                    continue

                elif user_input.lower() == "summary":
                    summary = agent.get_conversation_summary()
                    print("Convo summary:")
                    print("-" * 60)
                    print(summary)
                    print("-" * 60)
                    continue
                
                if not user_input:
                    continue
                
                print("Agent: ", end="", flush=True)
                response = agent.process_query(user_input)
                
                if debug_mode:
                    print(f"\n[DEBUG] Response length: {len(response)} characters")
                    print("[DEBUG] Full response:")
                    print("=" * 80)
                    print(response)
                    print("=" * 80)
                else:
                    print(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"Error: {e}")
                
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Make sure to set up your .env file with required API keys")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)