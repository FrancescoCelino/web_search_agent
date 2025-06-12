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

        # memory

        self.conversations = {}
        
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
        
        # Aggiungi un messaggio di sistema per aiutare con il contesto
        messages = state["messages"]
        
        # Se ci sono più di 1 messaggio, aggiungi contesto sempre
        if len(messages) > 1:
            # Crea un prompt di sistema più forte
            system_message = ("system", """You are a helpful AI assistant with access to web search, document retrieval, and SQL database tools.

IMPORTANT: You have access to the full conversation history. When users refer to previous topics, people, or information mentioned earlier (using words like "he", "she", "that person", "what was his name", "again", etc.), you MUST use the conversation context to understand what they're referring to.

DO NOT ask for clarification if the answer is clearly available in the conversation history. Instead, use the context to provide a direct answer.

Examples:
- If user asks "what was his name again?" and previously discussed "John Smith", answer "John Smith"
- If user asks "that company I mentioned" and previously discussed "Google", refer to Google
- Always prioritize conversation context over tool usage for reference questions""")
            
            # Inserisci il messaggio di sistema all'inizio se non c'è già
            if not any(isinstance(msg, tuple) and msg[0] == "system" for msg in messages):
                messages = [system_message] + messages
        
        response = self.llm_with_tools.invoke(messages)
        
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
            
            # Recupera la cronologia della conversazione
            if thread_id not in self.conversations:
                self.conversations[thread_id] = []
            
            # Controlla se la query fa riferimento a qualcosa di precedente
            conversation_history = self.conversations[thread_id].copy()
            
            # Se la query sembra fare riferimento a qualcosa di precedente, aggiungi contesto
            reference_keywords = ["his name", "her name", "that person", "the guy", "the girl", "what was", "who was", "the one", "that one", "again"]
            if any(keyword in query.lower() for keyword in reference_keywords) and conversation_history:
                # Crea un messaggio con contesto esplicito
                recent_exchanges = conversation_history[-4:]  # Ultimi 2 scambi
                context_info = []
                for role, content in recent_exchanges:
                    if role == "user":
                        context_info.append(f"User previously asked: {content}")
                    else:
                        context_info.append(f"Assistant responded: {content[:150]}...")
                
                context_text = "\n".join(context_info)
                enhanced_query = f"""Based on this recent conversation:
{context_text}

Current user question: {query}

Please answer the current question using the context from our conversation."""
                
                logger.info("ADDED CONVERSATION CONTEXT to query")
                query_to_process = enhanced_query
            else:
                query_to_process = query
            
            # Aggiungi il messaggio dell'utente alla cronologia  
            conversation_history.append(("user", query))
            
            # Stream the graph execution con la cronologia completa
            events = list(self.graph.stream(
                {"messages": conversation_history}, 
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
                        response_content = value["messages"][-1].content
                        logger.info(f"FINAL RESPONSE: {len(response_content)} characters")
                        
                        # Salva la risposta dell'assistente nella cronologia
                        self.conversations[thread_id].append(("user", query))  # Query originale, non modificata
                        self.conversations[thread_id].append(("assistant", response_content))
                        
                        # Mantieni solo gli ultimi 10 scambi (20 messaggi) per evitare overflow
                        if len(self.conversations[thread_id]) > 20:
                            self.conversations[thread_id] = self.conversations[thread_id][-20:]
                        
                        return response_content
            
            logger.warning("NO VALID RESPONSE GENERATED")
            return "I apologize, but I couldn't process your query properly."
            
        except Exception as e:
            logger.error(f"ERROR PROCESSING QUERY: {e}")
            return f"An error occurred while processing your query: {str(e)}"
        
    def get_conversation_history(self, thread_id: str = "default") -> list:
        """Get conversation history for a specific thread"""
        return self.conversations.get(thread_id, [])
    
    def clear_conversation(self, thread_id: str = "default"):
        """Clear conversation history for a specific thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            logger.info(f"Conversation history cleared for thread: {thread_id}")
    
    def get_conversation_summary(self, thread_id: str = "default") -> str:
        """Get a summary of the conversation"""
        history = self.get_conversation_history(thread_id)
        if not history:
            return "No conversation history"
        
        summary = f"Conversation with {len(history)} messages:\n"
        for i, (role, content) in enumerate(history[-6:]):  # Last 3 exchanges
            summary += f"{role}: {content[:100]}{'...' if len(content) > 100 else ''}\n"
        return summary
    
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