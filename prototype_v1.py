"""
Install dependencies:
    pip install chromadb sentence-transformers transformers torch

Run:
    python simple_rag_chroma.py
"""

import chromadb
import gradio as gr
from pathlib import Path
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# knowledge base
DATA_DIR = Path(__file__).parent / "data" / "sample_docs"

# load sample documents
DOCUMENTS = []
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
for filepath in DATA_DIR.iterdir():
    if filepath.is_file():
        content = filepath.read_text()
        chunks = splitter.split_text(content)
        DOCUMENTS.extend(chunks)

doc_ids = [f"doc{i}" for i in range(len(DOCUMENTS))]


# Set up vector database.
print("Setting up Chroma DB...")
client = chromadb.Client()
# client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
)
collection.add(documents=DOCUMENTS, ids=doc_ids)

# Retrieval
def retrieve(query: str, top_k: int = 2):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]  # list of matched document strings

# Generation
print("Loading generation model...")
gen_model_name = "google/flan-t5-small"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)


def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join(retrieved_docs)
    prompt = (
        f"Context: I am taking medication with the following information \n<medication_information>\n{context}\n</medication_information>\n"
        f"Question: {query}\n"
        f"Answer using only the context above:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    output_ids = gen_model.generate(**inputs, max_new_tokens=50)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def answer_query(query):
    retrieved = retrieve(query)

    print("retrieved docs:")
    for chunk in retrieved:
        print(" -", chunk.replace("\n", " "))

    answer = generate_answer(query, retrieved)
    return answer

user_interface = gr.Interface(
    fn=answer_query,
    inputs=gr.TextArea(label="User Query", placeholder="Ask a question about your mediciation"),
    outputs=gr.TextArea(label="Assistant Answer"),
    title="Personalized Medication Assistant"
)

# Gradio UI
# User input = query
# Output = RAG-grounded answer
if __name__ == "__main__":
    user_interface.launch()