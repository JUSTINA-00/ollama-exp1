import pandas as pd, ollama
from sentence_transformers import SentenceTransformer, util

products = pd.read_csv("products.csv")
faqs = pd.read_csv("faqs.csv")
reviews = pd.read_csv("reviews.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Combine columns into one text per row, then embed
products["text"] = products["name"] + ". " + products["description"]
product_emb = embedder.encode(products["text"].tolist())
faq_emb = embedder.encode(faqs["question"].tolist())

def llm(prompt):
    r = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"]

def search(query, df, emb, k=1):          # retrieval step of RAG
    scores = util.cos_sim(embedder.encode(query), emb)[0]
    return df.iloc[scores.argsort(descending=True)[:k].tolist()]

def faq_module(q):
    f = search(q, faqs, faq_emb).iloc[0]
    return llm(f"FAQ: {f.question} - {f.answer}\nAnswer this customer briefly using the FAQ: {q}")

def recommend_module(q):
    items = search(q, products, product_emb, k=3)
    return llm(f"Products:\n{items.text.to_string(index=False)}\nRecommend the best for: {q}. Keep it short.")

def product_module(q):
    p = search(q, products, product_emb).iloc[0]
    return llm(f"Product: {p.text}. Price: Rs.{p.price}\nAnswer briefly using only this info: {q}")

def review_module(q):
    p = search(q, products, product_emb).iloc[0]
    r = reviews[reviews.product_id == p.id]
    return f"{p['name']} - Avg rating: {r.rating.mean():.1f}/5\n" + \
           llm(f"Summarize these reviews in 3 lines with overall sentiment:\n{r.review.to_string(index=False)}")

def intent_module(q):
    out = llm(f"Classify into one word: faq, recommend, product or review.\nQuery: {q}\nOnly one word.").lower()
    return next((w for w in ["recommend", "product", "review"] if w in out), "faq")

def chatbot(q):
    intent = intent_module(q)
    print("[Intent:", intent + "]")
    return {"faq": faq_module, "recommend": recommend_module,
            "product": product_module, "review": review_module}[intent](q)

while True:
    q = input("\nYou: ")
    if q == "exit": break
    print("Bot:", chatbot(q))
