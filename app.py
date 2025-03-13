import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
import openai
import uvicorn
import logging

from brain import create_or_load_index, update_index
from utils import detect_docs_changes, load_checksums, save_checksums, get_mdx_files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()

# Verify if the necessary environment variables are set
required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
]
missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure OpenAI setup
logger.info("Setting up Azure OpenAI...")
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
logger.info("Azure OpenAI setup completed.")


# Load MDX files and create vector database
docs_folder = os.path.join(os.getcwd(), "docs")
mdx_file_paths = get_mdx_files(docs_folder)

if not mdx_file_paths:
    logger.error("No MDX files found in the docs folder.")
    sys.exit(1)

mdx_files = [open(f, "rb").read() for f in mdx_file_paths]
mdx_file_names = [os.path.basename(f) for f in mdx_file_paths]


# Checksums file to keep track of changes in documentation
checksum_file = os.path.join(os.getcwd(), "checksums.pkl")
last_checksums = load_checksums(checksum_file)
has_changes, current_checksums = detect_docs_changes(mdx_file_paths, last_checksums)

if has_changes:
    logger.info("Changes detected. Updating vector database...")
    vectordb = create_or_load_index(mdx_files, mdx_file_names)
    update_index(vectordb, mdx_files, mdx_file_names)
    save_checksums(current_checksums, checksum_file)
else:
    logger.info("No changes detected. Loading existing vector database...")
    vectordb = create_or_load_index(mdx_files, mdx_file_names)

logger.info("Creating conversational chain...")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    max_messages=10,
)

template = """
    You are a helpful assistant specialized in answering technical questions related to Keploy. You are provided with context from a vector database and a chat history. Your task is to answer the user's question based on the provided context and the chat history. If you don't know the answer, just say 'I don't know'. Do not try to make up an answer. If the question is not related to Keploy, say 'I am not sure about that'."

    Context: {context}
    Question: {question}
    Answer:

    """

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
api_key = os.getenv("AZURE_OPENAI_API_KEY")
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o-global-standard",
    api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_type="azure",
    temperature=0.7,
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt},
)
logger.info("Conversational chain created successfully.")


# Define the Question model for the API request
class Question(BaseModel):
    question: str


# API endpoint to handle chat queries
@app.post("/chat")
def chat(question: Question):
    logger.info("Received chat request")
    if not question.question:
        logger.warning("No question provided")
        raise HTTPException(status_code=400, detail="No question provided")

    try:
        # Perform similarity search on vector database
        search_results = vectordb.similarity_search(question.question, k=3)
        context = "\n".join([doc.page_content for doc in search_results])

        # Get response from conversation chain
        response = conversation_chain({"question": question.question})

        # Log the response for debugging
        logger.info(f"Response from conversation chain: {response}")

        # Prepare the result response
        result = {
            "answer": response["answer"],
            "sources": [
                doc.metadata.get("source", "Unknown")
                for doc in response.get("source_documents", [])
            ],
        }
        return result

    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred during chat processing"
        )


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    logger.info("Starting FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI app started.")
