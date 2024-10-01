import sys
print("Starting import process...")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import uvicorn
print("Imports completed successfully.")

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv()
print("Environment variables loaded.")

app = FastAPI(__name__)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Setting up Azure OpenAI...")
# Azure OpenAI setup
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
print("Azure OpenAI setup completed.")

# Function to get all MDX files from the docs directory and its subdirectories
def get_mdx_files(directory):
    print(f"Searching for MDX files in {directory}")
    mdx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                mdx_files.append(os.path.join(root, file))
    print(f"Found {len(mdx_files)} MDX files.")
    return mdx_files

# Function to create a vectordb for the provided MDX files
def create_vectordb(files, filenames):
    print("Creating vectordb...")
    # Import the function here to avoid circular imports
    from brain import get_index_for_mdx
    vectordb = get_index_for_mdx(files, filenames)
    print("Vectordb created successfully.")
    return vectordb

print("Loading MDX files...")
# Load MDX files and create vectordb
docs_folder = os.path.join(os.getcwd(), "docs")
mdx_file_paths = get_mdx_files(docs_folder)

if mdx_file_paths:
    print("Creating vectordb from MDX files...")
    mdx_files = [open(f, "rb").read() for f in mdx_file_paths]
    mdx_file_names = [os.path.basename(f) for f in mdx_file_paths]
    vectordb = create_vectordb(mdx_files, mdx_file_names)

    print("Creating conversational chain...")
    # Create a conversational chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer", max_messages=10)

    
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="keploy-gpt4o",
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        openai_api_type="azure",
        temperature=0.7
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    print("Conversational chain created successfully.")
else:
    print("Error: No MDX files found in the docs folder.")
    sys.exit(1)



class Question(BaseModel):
    question: str


@app.post('/chat')
def chat(question: Question):
    print("Received chat request")
    if not question.question:
        raise HTTPException(status_code=400, detail="No question provided")

    # Perform similarity search
    
    try:
        search_results = vectordb.similarity_search(question.question, k=3)
        context = "\n".join([doc.page_content for doc in search_results])

        # Get response from conversation chain
        response = conversation_chain({"question": question.question})

        # Prepare the response
        result = {
            "answer": response['answer'],
            "sources": [doc.metadata['source'] for doc in response.get('source_documents', [])]
        }

        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    print("Starting fastapi app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("fastapi app started.")
