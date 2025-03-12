import os
from io import BytesIO
from typing import Tuple, List

from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.indexes import SQLRecordManager, index

record_manager = SQLRecordManager(
    "faiss/document_index", db_url="sqlite:///record_manager_cache.sql"
)


def parse_mdx(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    content = file.read().decode('utf-8')
    # You might want to add more sophisticated MDX parsing here
    # print("Parsing MDX file:", filename)
    return [content], filename 


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    doc_chunks = []
    for i, page in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(page)
        for j, chunk in enumerate(chunks):
            # Calculate the start and end line numbers for this chunk
            start_line = page[:page.index(chunk)].count('\n') + 1
            end_line = start_line + chunk.count('\n')

            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk": j,
                    "source": f"{filename}:{start_line}-{end_line}",
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line
                }
            )
            doc_chunks.append(doc)
    return doc_chunks


def get_index_for_mdx(mdx_files, mdx_names):
    record_manager.create_schema()
    print("Updating index for MDX files...")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="keploy-docs-embedding",
        chunk_size=1,  # You can adjust this value as needed
    )

    documents = []
    for mdx_file, mdx_name in zip(mdx_files, mdx_names):
        text, filename = parse_mdx(BytesIO(mdx_file), mdx_name)
        # print("Text to docs:", filename)
        documents = documents + text_to_docs(text, filename)

    if os.path.exists("document_index"):
        vector_store = FAISS.load_local(
            folder_path="document_index",
            embeddings=AzureOpenAIEmbeddings(model="keploy-docs-embedding"),
            allow_dangerous_deserialization=True
        )
        index(documents, record_manager, vector_store,
              cleanup="full", source_id_key="source")
    else:
        vector_store = FAISS(embedding_function=embeddings)
        index(documents, record_manager, vector_store,
              cleanup="full", source_id_key="source")
        vector_store.save_local("document_index")

    return vector_store
