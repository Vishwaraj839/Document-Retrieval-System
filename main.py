from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained transformer model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index for similarity search
embedding_dimension = 384  # Dimension of 'all-MiniLM-L6-v2' model embeddings
index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance (can be replaced with cosine similarity)

# In-memory document store for simplicity (in production, this would be a DB)
documents = [
    "Artificial intelligence has transformed healthcare by enabling faster diagnoses, improving patient outcomes, and reducing costs.",
    "Machine learning models assist in predicting diseases, while computer vision techniques enhance medical imaging.",
    "AI applications in medicine include diagnostics, drug discovery, and personalized treatments.",
    "AI algorithms help doctors in making data-driven decisions, resulting in improved precision."
]

# Encode documents and add them to FAISS index
doc_embeddings = model.encode(documents, convert_to_tensor=False)
index.add(np.array(doc_embeddings, dtype=np.float32))

# API request model
class SearchRequest(BaseModel):
    text: str
    top_k: int = 3  # Number of top results to fetch
    threshold: float = 0.75  # Threshold for similarity

# Function to retrieve top-k similar documents
def retrieve_top_k_results(query_embedding, top_k, threshold):
    # Perform similarity search with FAISS
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    # Filter by threshold (distances are L2, so lower is better)
    top_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:
            top_results.append(documents[idx])
    
    return top_results

# API endpoint to search documents and generate context
@app.post("/search")
def search_documents(request: SearchRequest):
    # Encode the query text to an embedding
    query_embedding = model.encode(request.text, convert_to_tensor=False)
    
    # Retrieve top-k similar documents
    top_k_docs = retrieve_top_k_results(query_embedding, request.top_k, request.threshold)
    
    # If no documents found, raise an error
    if not top_k_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Aggregate context (concatenate the retrieved document texts)
    context = " ".join(top_k_docs)
    
    # Return the generated context
    return {"context": context, "documents": top_k_docs}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"message": "API is active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
