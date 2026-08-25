import chromadb
import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set up vector database.
client = chromadb.PersistentClient(path="./medication_db")
collection = client.get_collection(name="knowledge_base")

# Retrieval
def retrieve(query: str, top_k: int = 5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]  # list of matched document strings

# Generation
print("Loading generation model...")
gen_model_name = "google/flan-t5-base"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)


def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join(retrieved_docs)
    prompt = (
        f"Context: I am taking medication with the following information \n{context}\n\n"
        f"Question: {query}\n"
        f"Answer the following question using only the context above:"
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
    inputs=gr.TextArea(label="User Query", placeholder="Ask a question about your medication"),
    outputs=gr.TextArea(label="Assistant Answer"),
    title="Personalized Medication Assistant"
)

# Gradio UI
# User input = query
# Output = RAG-grounded answer
if __name__ == "__main__":
    user_interface.launch()
    # print(answer_query("What are the most common side effects of the medication I am taking?"))