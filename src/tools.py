"""
Tool definitions for AI Agent
"""
import logging
from typing import Optional
from operator import itemgetter

from langchain.tools import tool
from langchain.chains import create_sql_query_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel

# Import corretto per embeddings (fix deprecation warning)
try:
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except ImportError:
    # Fallback se langchain-huggingface non è installato
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from .config import Config
from .utils import load_documents, split_documents, clean_sql_query

logger = logging.getLogger(__name__)

# Pydantic schemas for tools
class RagToolSchema(BaseModel):
    question: str

class SQLToolSchema(BaseModel):
    question: str

class ToolManager:
    """Manages all tools for the AI Agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.vectorstore = None
        self.db = None
        self._setup_tools()
    
    def _setup_tools(self):
        """Initialize all tools"""
        try:
            self._setup_vectorstore()
            self._setup_database()
            logger.info("All tools initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up tools: {e}")
    
    def _setup_vectorstore(self):
        """Setup vector store for document retrieval"""
        try:
            # Load and process documents
            documents = load_documents(Config.DOCUMENTS_FOLDER)
            if not documents:
                logger.warning("No documents found for vector store")
                return
            
            splits = split_documents(
                documents, 
                Config.CHUNK_SIZE, 
                Config.CHUNK_OVERLAP
            )
            
            # Create embeddings
            embedding_function = SentenceTransformerEmbeddings(
                model_name=Config.EMBEDDING_MODEL
            )
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                collection_name=Config.COLLECTION_NAME,
                documents=splits,
                embedding=embedding_function,
                persist_directory=Config.CHROMA_DB_PATH
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
    
    def _setup_database(self):
        """Setup SQL database connection"""
        try:
            self.db = SQLDatabase.from_uri(Config.DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def get_web_search_tool(self):
        """Get web search tool"""
        try:
            # Tool semplice senza logging wrapper per ora
            web_tool = TavilySearchResults(max_results=Config.MAX_SEARCH_RESULTS)
            logger.info("Web search tool created successfully")
            return web_tool
            
        except Exception as e:
            logger.error(f"Error creating web search tool: {e}")
            return None
    
    def get_retriever_tool(self):
        """Get document retriever tool"""
        @tool(args_schema=RagToolSchema)
        def retriever_tool(question: str) -> str:
            """Tool to Retrieve Semantically Similar documents to answer User Questions related to FutureSmart AI"""
            logger.info(f"DOCUMENT RETRIEVAL TOOL CALLED with question: {question}")
            
            if not self.vectorstore:
                logger.warning("DOCUMENT RETRIEVAL: No vectorstore available")
                return "Document retrieval not available - no documents loaded"
            
            try:
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": Config.RETRIEVAL_K}
                )
                retriever_result = retriever.invoke(question)
                
                if not retriever_result:
                    logger.info("DOCUMENT RETRIEVAL: No relevant documents found")
                    return "No relevant documents found"
                
                result = "\n\n".join(doc.page_content for doc in retriever_result)
                logger.info(f"DOCUMENT RETRIEVAL RESULT: {len(result)} characters, {len(retriever_result)} documents")
                return result
                
            except Exception as e:
                logger.error(f"DOCUMENT RETRIEVAL ERROR: {e}")
                return f"Error retrieving documents: {str(e)}"
        
        return retriever_tool
    
    def get_nl2sql_tool(self):
        """Get NL to SQL tool"""
        @tool(args_schema=SQLToolSchema)
        def nl2sql_tool(question: str) -> str:
            """Tool to Generate and Execute SQL Query to answer User Questions related to chinook DB"""
            logger.info(f"SQL TOOL CALLED with question: {question}")
            
            if not self.db:
                logger.warning("SQL TOOL: No database available")
                return "SQL database not available"
            
            try:
                execute_query = QuerySQLDatabaseTool(db=self.db)
                write_query = create_sql_query_chain(self.llm, self.db)

                chain = (
                    RunnablePassthrough.assign(
                        query=write_query | RunnableLambda(clean_sql_query)
                    ).assign(
                        result=itemgetter("query") | execute_query
                    )
                )

                response = chain.invoke({"question": question})
                logger.info(f"SQL TOOL RESULT: {len(str(response['result']))} characters")
                return response['result']
                
            except Exception as e:
                logger.error(f"SQL TOOL ERROR: {e}")
                return f"Error executing SQL query: {str(e)}"
        
        return nl2sql_tool
    
    def get_all_tools(self):
        """Get all available tools"""
        tools = []
        
        # Web search tool
        try:
            web_tool = self.get_web_search_tool()
            if web_tool:
                tools.append(web_tool)
                logger.info("Web search tool available")
            else:
                logger.warning("Web search tool not available")
        except Exception as e:
            logger.error(f"Web search tool error: {e}")
        
        # Document retrieval tool
        try:
            if self.vectorstore:
                tools.append(self.get_retriever_tool())
                logger.info("Document retrieval tool available")
            else:
                logger.warning("Document retrieval tool not available - no vectorstore")
        except Exception as e:
            logger.error(f"Document retrieval tool error: {e}")
        
        # SQL tool
        try:
            if self.db:
                tools.append(self.get_nl2sql_tool())
                logger.info("SQL tool available")
            else:
                logger.warning("SQL tool not available - no database")
        except Exception as e:
            logger.error(f"SQL tool error: {e}")
        
        logger.info(f"Total available tools: {len(tools)}")
        return tools