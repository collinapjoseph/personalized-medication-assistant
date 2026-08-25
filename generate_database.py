import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# knowledge base
DATA_DIR = Path(__file__).parent / "data" / "sample_docs_1"

# load sample documents
DOCUMENTS = []
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10)
for filepath in DATA_DIR.iterdir():
    if filepath.is_file():
        content = filepath.read_text()
        chunks = splitter.split_text(content)
        DOCUMENTS.extend(chunks)

doc_ids = [f"doc{i}" for i in range(len(DOCUMENTS))]

# Set up vector database.
client = chromadb.PersistentClient(path="./medication_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
)
collection.add(documents=DOCUMENTS, ids=doc_ids)
