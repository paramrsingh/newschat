"""
Param Singh 5/1/2025
Mistral RAG and Web Search demo
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Any, Tuple
from dotenv import load_dotenv
import gradio as gr
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try imports with error handling
try:
    from llama_index.embeddings.mistralai import MistralAIEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.mistralai import MistralAI
    from llama_index.tools.tavily_research.base import TavilyToolSpec
    from llama_index.core.tools import FunctionTool
    from llama_index.core.settings import Settings
    from llama_index.core import load_index_from_storage, StorageContext
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    print(
        f"Error: Missing required libraries. Please run: pip install llama-index-embeddings-mistralai llama-index-vector-stores-faiss llama-index-llms-mistralai llama-index-tools-tavily")
    sys.exit(1)


def load_environment_variables() -> Tuple[Optional[str], Optional[str]]:
    """Load environment variables """
    try:
        load_dotenv()
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")

        if not tavily_api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables")
        if not mistral_api_key:
            logger.warning("MISTRAL_API_KEY not found in environment variables")

        return tavily_api_key, mistral_api_key
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return None, None


def initialize_models(mistral_api_key: str) -> Tuple[Optional[MistralAI], Optional[MistralAIEmbedding]]:
    """Initialize LLM and embedding models"""
    try:
        llm = MistralAI(api_key=mistral_api_key, model='mistral-large-latest')
        embed_model = MistralAIEmbedding(
            model_name='mistral-embed',
            api_key=mistral_api_key,
            embed_batch_size=6
        )
        return llm, embed_model
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return None, None


def load_vector_index(storage_path: str = "./storage") -> Optional[Any]:
    """Load persisted FAISS index"""
    try:
        vector_store = FaissVectorStore.from_persist_dir(storage_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=storage_path
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except FileNotFoundError:
        logger.error(f"Storage directory not found at {storage_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load vector index: {e}")
        return None


def custom_internet_search(query: str, tavily_tool: TavilyToolSpec, llm: MistralAI) -> str:
    """
    Function to handle using the Tavily tool to get the URLs and raw text from an internet search.

    :param query: User query
    :param tavily_tool: Initialized Tavily tool
    :param llm: Initialized Mistral LLM
    :return: Result of the internet search or error message
    """
    if not query or not tavily_tool or not llm:
        return "Error: Missing query or search tools. Please try again."

    try:
        # Attempt to get results with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = tavily_tool.search(query, max_results=10)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Search attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)  # Add a small delay before retry
                else:
                    logger.error(f"Search failed after {max_retries} attempts: {e}")
                    return f"Sorry, I couldn't search the internet right now. Please try again later."

        if not results or len(results) == 0:
            logger.warning(f"No search results found for query: {query}")
            return "I couldn't find any relevant information for your search query."

        prompt_context_str = ""

        for result in results:
            url = result.metadata.get('url', 'No URL available')
            text = result.text if hasattr(result, 'text') else 'No text available'
            prompt_context_str += f"url: {url}\n"
            prompt_context_str += f"text: {text}\n"

        messages = [
            ChatMessage(role="system", content="You are a news expert. \n Using the provided urls and text in the context  \
                                              to best answer the query in a few sentences but not longer than a paragraph. \n List the provided urls if you used that text content in your response"),
            ChatMessage(role="user", content=f"Given the provided context, Query:{query} \
                                              Context:{prompt_context_str}"),
        ]

        response = llm.chat(messages)
        return response.message.content
    except Exception as e:
        logger.error(f"Error in internet search: {str(e)}")
        return f"An error occurred while searching the internet: {str(e)}. Please try again later."


def custom_rag(query: str, index: Any, llm: MistralAI) -> str:
    """
    Function to handle using the stored FAISS index as a LlamaIndex QueryEngine.

    :param query: User query
    :param index: Loaded index
    :param llm: Initialized Mistral LLM
    :return: Result of the RAG search or error message
    """
    if not query or not index or not llm:
        return "Error: Missing query or RAG components. Please try again."

    try:
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query)

        sources = ''
        if hasattr(response, 'metadata') and response.metadata:
            for i in response.metadata.items():
                if isinstance(i, tuple) and len(i) > 1 and isinstance(i[1], dict):
                    file_path = i[1].get('file_path', 'Unknown source')
                    sources += f"{file_path}\n"

        response_str = response.response + " \nSources cited: \n" + sources
        return response_str
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        return f"An error occurred while searching the BBC archive: {str(e)}. Please try again later."


def create_tools(tavily_tool: TavilyToolSpec, index: Any, llm: MistralAI):
    """Create tools with necessary dependencies injected"""

    # Wrap the internet search function to include the dependencies
    def internet_search_wrapper(query: str) -> str:
        return custom_internet_search(query, tavily_tool, llm)

    # Wrap the RAG function to include the dependencies
    def rag_wrapper(query: str) -> str:
        return custom_rag(query, index, llm)

    # Tool definitions
    web_search_tool = FunctionTool.from_defaults(
        fn=internet_search_wrapper,
        name="internet_search",
        description="Useful for getting the most recent news or searching the internet"
    )

    rag_tool = FunctionTool.from_defaults(
        fn=rag_wrapper,
        name="rag_bbc_archive_search",
        description="Useful for when the query is for BBC archive"
    )

    return [web_search_tool, rag_tool]


def main_chat(user_query: str, history=None, llm=None, tools=None) -> str:
    """Main chat function with error handling"""
    if not user_query:
        return "Please enter a question."

    if not llm or not tools:
        return "System error: LLM or tools not properly initialized. Please restart the application."

    try:
        response = llm.predict_and_call(tools=tools, verbose=False, user_msg=user_query)
        return response.response
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        return f"Sorry, I encountered an error processing your request: {str(e)}. Please try again."


# Main application
def initialize_app():
    """Initialize the application with all necessary components"""

    # Load environment variables
    tavily_api_key, mistral_api_key = load_environment_variables()
    if not tavily_api_key or not mistral_api_key:
        error_msg = "Missing API keys. Please check your .env file."
        logger.error(error_msg)
        return gr.ChatInterface(lambda x, y: error_msg, title="Error")

    # Initialize models
    llm, embed_model = initialize_models(mistral_api_key)
    if not llm or not embed_model:
        error_msg = "Failed to initialize models. Check logs for details."
        logger.error(error_msg)
        return gr.ChatInterface(lambda x, y: error_msg, title="Error")

    # Set global settings
    try:
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 2048
        Settings.chunk_overlap = 128
    except Exception as e:
        logger.error(f"Failed to set global settings: {e}")

    # Initialize Tavily tool
    try:
        tavily_tool = TavilyToolSpec(api_key=tavily_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily tool: {e}")
        error_msg = "Failed to initialize search tool. Check logs for details."
        return gr.ChatInterface(lambda x, y: error_msg, title="Error")

    # Load vector index
    index = load_vector_index()
    if not index:
        error_msg = "Failed to load vector index. Make sure the storage directory exists and is properly populated."
        logger.error(error_msg)
        return gr.ChatInterface(lambda x, y: error_msg, title="Error")

    # Create tools
    tools = create_tools(tavily_tool, index, llm)

    # Create a closure for the chat function to include dependencies
    def chat_with_dependencies(user_query: str, history=None):
        return main_chat(user_query, history, llm, tools)

    # Create Gradio UI
    app = gr.ChatInterface(
        fn=chat_with_dependencies,
        chatbot=gr.Chatbot(label="News Chatbot"),
        textbox=gr.Textbox(
            placeholder="Ask your news question here.",
            container=True,
            autofocus=True,
        ),
        title="News Chat with Mistral",
        description=("Ask questions about the news, or BBC archive"),
        theme="ocean",
        examples=[
            ["Use the BBC archive to tell me about Jean-Michel Jarre and the Parken concert"],
            ["Check the internet for the latest news on the selection of the new pope"],
            ["Give me the latest on the NY Mets and who they are playing next"]
        ],
    )

    return app


# Gradio UI bits
if __name__ == "__main__":
    try:
        app = initialize_app()
        app.launch(share=False)
    except Exception as e:
        logger.critical(f"Failed to launch application: {e}")
        print(f"Critical error: {e}")
        sys.exit(1)