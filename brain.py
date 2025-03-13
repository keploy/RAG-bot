import os
from io import BytesIO
from typing import List

from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS


def parse_mdx(file: BytesIO, filename: str) -> List[Document]:
    content = file.read().decode("utf-8")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    documents = []
    for i, chunk in enumerate(chunks):
        start_line = content[: content.index(chunk)].count("\n") + 1
        end_line = start_line + chunk.count("\n")
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk": i,
                "source": f"{filename}:{start_line}-{end_line}",
                "filename": filename,
                "start_line": start_line,
                "end_line": end_line,
            },
        )
        documents.append(doc)
    return documents


def create_or_load_index(mdx_files, mdx_names):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="keploy-docs-embedding",
        chunk_size=1,  # You can adjust this value as needed
    )

    # Load existing index if document_index folder exists
    if os.path.exists("document_index"):
        return FAISS.load_local(
            folder_path="document_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Creating new FAISS index...")
    documents = []
    for mdx_file, mdx_name in zip(mdx_files, mdx_names):
        docs = parse_mdx(BytesIO(mdx_file), mdx_name)
        documents.extend(docs)

    index = FAISS.from_documents(documents, embeddings)
    index.save_local("document_index")
    print("New FAISS index created and saved.")
    return index


def update_index(index: FAISS, mdx_files, mdx_names):
    documents = []
    for mdx_file, mdx_name in zip(mdx_files, mdx_names):
        docs = parse_mdx(BytesIO(mdx_file), mdx_name)
        documents.extend(docs)

    index.add_documents(documents)
    index.save_local("document_index")
    print("FAISS index updated with new documents.")
