import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# knowledge base
DATA_DIR = Path(__file__).parent / "data" / "sample_docs_1"
DB_PATH = Path(__file__).parent / "medication_db"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 10
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# load sample documents
DOCUMENTS = []
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
for filepath in DATA_DIR.iterdir():
    if filepath.is_file():
        content = filepath.read_text()
        chunks = splitter.split_text(content)
        DOCUMENTS.extend(chunks)

doc_ids = [f"doc{i}" for i in range(len(DOCUMENTS))]

# Set up vector database.
client = chromadb.PersistentClient(path=DB_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)
collection.add(documents=DOCUMENTS, ids=doc_ids)
