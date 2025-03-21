import os
from io import BytesIO
from typing import Tuple, List, Dict
import re
from datetime import datetime
from pathlib import Path
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores.faiss import FAISS

def extract_code_blocks(content: str) -> List[Dict]:
    """Extract code blocks from markdown content."""
    code_pattern = r"```(\w+)?\n(.*?)\n```"
    code_blocks = []

    for match in re.finditer(code_pattern, content, re.DOTALL):
        language = match.group(1) or "text"
        code = match.group(2)
        code_blocks.append({
            "language": language,
            "code": code.strip()
        })

    return code_blocks

def parse_mdx(file: BytesIO, filename: str) -> Tuple[List[str], str, List[Dict]]:
    content = file.read().decode('utf-8')
    code_blocks = extract_code_blocks(content)

    # removing code blocks from content to avoid duplicating
    clean_content = re.sub(r"```(\w+)?\n.*?\n```", "", content, flags=re.DOTALL)

    return [clean_content], filename, code_blocks

def text_to_docs(text: List[str], filename: str, code_blocks: List[Dict] = None) -> List[Document]:
    if isinstance(text, str):
        text = [text]

    doc_chunks = []

    # textual content processing
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )

    for i, page in enumerate(text):
        # first split by headers
        header_splits = markdown_splitter.split_text(page)

        for split in header_splits:
            chunks = text_splitter.split_text(split.page_content)
            for j, chunk in enumerate(chunks):
                start_line = page[:page.index(chunk)].count('\n') + 1
                end_line = start_line + chunk.count('\n')

                metadata = {
                    "chunk": j,
                    "source": f"{filename}:{start_line}-{end_line}",
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line,
                    "type": "text",
                    "header": split.metadata.get("Header 1", "") or split.metadata.get("Header 2", "") or split.metadata.get("Header 3", ""),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(filename)).isoformat() if os.path.exists(filename) else None
                }

                doc = Document(page_content=chunk, metadata=metadata)
                doc_chunks.append(doc)

    # Processing of code blocks,if any present
    if code_blocks:
        for i, block in enumerate(code_blocks):
            doc = Document(
                page_content=block["code"],
                metadata={
                    "chunk": f"code_{i}",
                    "source": f"{filename}:code_block_{i}",
                    "filename": filename,
                    "type": "code",
                    "language": block["language"],
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(filename)).isoformat() if os.path.exists(filename) else None
                }
            )
            doc_chunks.append(doc)

    return doc_chunks

def get_index_for_mdx(mdx_files, mdx_names, force_refresh=False):
    print("Creating/updating index for MDX files...")

    index_path = "document_index"
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="keploy-docs-embedding",
        chunk_size=1,
    )

    # loading existing index if it exists and no force refresh
    if os.path.exists(index_path) and not force_refresh:
        existing_index = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        # check if files modified
        modified_files = []
        for mdx_file, mdx_name in zip(mdx_files, mdx_names):
            if isinstance(mdx_file, bytes):
                continue

            file_path = Path(mdx_name)
            if file_path.exists():
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

                # check if file modified since last indexing
                existing_docs = [doc for doc in existing_index.docstore._dict.values()
                               if doc.metadata["filename"] == mdx_name]

                if not existing_docs or any(doc.metadata.get("last_modified", "") != last_modified
                                          for doc in existing_docs):
                    modified_files.append((mdx_file, mdx_name))

        if not modified_files:
            print("No modified files found, using existing index.")
            return existing_index

        # Processing only modified files
        print(f"Processing {len(modified_files)} modified files...")
        documents = []
        for mdx_file, mdx_name in modified_files:
            text, filename, code_blocks = parse_mdx(BytesIO(mdx_file), mdx_name)
            documents.extend(text_to_docs(text, filename, code_blocks))

        # update existing index
        existing_index.add_documents(documents)
        existing_index.save_local(index_path)
        return existing_index

    print("Creating new index...")
    documents = []
    for mdx_file, mdx_name in zip(mdx_files, mdx_names):
        text, filename, code_blocks = parse_mdx(BytesIO(mdx_file), mdx_name)
        documents.extend(text_to_docs(text, filename, code_blocks))

    index = FAISS.from_documents(documents, embeddings)
    index.save_local(index_path)
    return index