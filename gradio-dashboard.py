import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

# ============== 1. Environment ==============
load_dotenv()

# ============== 2. Load Book Dataset ==============
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# ============== 3. Load & Split Descriptions ==============
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Filter empty/short chunks
documents = [doc for doc in documents if doc.page_content.strip() and len(doc.page_content) > 50]

# For testing, limit to first 200 chunks (safe)
documents = documents[:200]

# ============== 4. Embedding Model ==============
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Use 'cuda' if you have a GPU
)

# ============== 5. Use Persistent Vector DB ==============
db_path = "chroma_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

# Avoid re-embedding if DB already exists
try:
    db_books = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function
    )
    if len(db_books.get()['documents']) == 0:
        raise ValueError("DB is empty, re-embedding.")
except Exception:
    db_books = Chroma.from_documents(
        documents,
        embedding=embedding_function,
        persist_directory=db_path
    )

# ============== 6. Recommender Logic ==============
def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs if rec.page_content.strip()]

    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row.get("description", "No description"))
        short_desc = " ".join(description.split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            author_str = authors[0]

        caption = f"{row['title']} by {author_str}: {short_desc}"
        results.append((row["large_thumbnail"], caption))
    return results

# ============== 7. Gradio UI ==============
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Choose a category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Choose an emotion", value="All")
        submit_button = gr.Button("Get Recommendations")

    gr.Markdown("## 🔍 Results")
    output_gallery = gr.Gallery(label="Recommended Books", columns=4, rows=4)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output_gallery)

# ============== 8. Run App ==============
if __name__ == "__main__":
    dashboard.launch()
