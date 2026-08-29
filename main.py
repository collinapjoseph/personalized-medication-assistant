import os
import chromadb
import gradio as gr
from google import genai
from pathlib import Path

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.7-flash"
DB_PATH = Path(__file__).parent / "medication_db"
COLLECTION_NAME = "knowledge_base"

# Set up vector database.
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Retrieval
def retrieve(query: str, top_k: int = 5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]  # list of matched document strings

# Generation
print("Loading generation model...")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join(retrieved_docs)
    prompt = (
        f"Context: I am taking medication with the following information \n{context}\n\n"
        f"Question: {query}\n"
        f"Answer the following question using only the context above:"
    )

    res = client_gemini.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    return res.output_text

def answer_query(query):
    retrieved = retrieve(query)

    print("retrieved docs:")
    for chunk in retrieved:
        print(" -", chunk.replace("\n", " "))

    answer = generate_answer(query, retrieved)
    return answer

user_interface = gr.Interface(
    fn=answer_query,
    inputs=gr.TextArea(label="User Query", placeholder="Ask a question about your medication"),
    outputs=gr.TextArea(label="Assistant Answer"),
    title="Personalized Medication Assistant"
)

# Gradio UI
# User input = query
# Output = RAG-grounded answer
if __name__ == "__main__":
    user_interface.launch()
