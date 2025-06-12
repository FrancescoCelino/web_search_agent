"""
AI Agent core implementation
"""
import logging
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from .config import Config
from .tools import ToolManager

logger = logging.getLogger(__name__)

class State(TypedDict):
    """State definition for the agent graph"""
    messages: Annotated[list, add_messages]

class AIAgent:
    """Main AI Agent class that orchestrates all tools"""
    
    def __init__(self):
        """Initialize the AI Agent"""
        # Validate configuration
        Config.validate()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=Config.MODEL_NAME)
        
        # Initialize tools
        self.tool_manager = ToolManager(self.llm)
        self.tools = self.tool_manager.get_all_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("AI Agent initialized successfully")
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Create graph builder
        graph_builder = StateGraph(State)
        
        # Add chatbot node
        graph_builder.add_node("chatbot", self._chatbot_node)
        
        # Add tools node
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        
        # Set entry point
        graph_builder.set_entry_point("chatbot")
        
        # Compile graph
        return graph_builder.compile()
    
    def _chatbot_node(self, state: State):
        """Chatbot node function"""
        logger.info("CHATBOT NODE: Processing user message")
        response = self.llm_with_tools.invoke(state["messages"])
        
        # Logging sicuro senza assumere attributi specifici
        logger.info(f"CHATBOT NODE: Response type: {type(response).__name__}")
        
        return {"messages": [response]}
    
    def process_query(self, query: str, thread_id: str = "default") -> str:
        """
        Process a user query and return the response
        
        Args:
            query (str): User query
            thread_id (str): Thread ID for conversation continuity
            
        Returns:
            str: Agent response
        """
        try:
            logger.info(f"PROCESSING QUERY: '{query}'")
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream the graph execution e conta gli eventi
            events = list(self.graph.stream(
                {"messages": [("user", query)]}, 
                config
            ))
            
            logger.info(f"GRAPH EXECUTION: {len(events)} events processed")
            
            # Log di ogni evento per debugging
            for i, event in enumerate(events):
                logger.debug(f"Event {i+1}: {list(event.keys())}")
                
                # Se l'evento contiene "tools", significa che i tool sono stati chiamati
                if "tools" in event:
                    logger.info("TOOLS NODE EXECUTED: Tools were actually called!")
                elif "chatbot" in event:
                    logger.info("CHATBOT NODE EXECUTED")
            
            # Get the last response
            if events:
                last_event = events[-1]
                for value in last_event.values():
                    if "messages" in value and value["messages"]:
                        response = value["messages"][-1].content
                        logger.info(f"FINAL RESPONSE: {len(response)} characters")
                        return response
            
            logger.warning("NO VALID RESPONSE GENERATED")
            return "I apologize, but I couldn't process your query properly."
            
        except Exception as e:
            logger.error(f"ERROR PROCESSING QUERY: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def get_available_tools(self) -> list:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]
    
    def get_graph_visualization(self):
        """Get graph visualization (requires additional dependencies)"""
        try:
            from IPython.display import Image
            return Image(self.graph.get_graph().draw_mermaid_png())
        except ImportError:
            logger.warning("IPython not available for graph visualization")
            return None
        except Exception as e:
            logger.error(f"Error generating graph visualization: {e}")
            return None