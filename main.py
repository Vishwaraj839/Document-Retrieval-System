import redis
import json
from fastapi import FastAPI, HTTPException, Depends
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from pydantic import BaseModel
import time
import logging
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session

# PostgreSQL database 
DATABASE_URL = "postgresql://fastapi_user:user123@localhost/trademarkia_db"

# Set up the SQLAlchemy 
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


class UserRequest(Base):
    __tablename__ = "user_requests"
    user_id = Column(String, primary_key=True, index=True)
    request_count = Column(Integer, default=1)


Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained transformer model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index for similarity search
embedding_dimension = 384  
index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance 

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

# Initialize Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)


class SearchRequest(BaseModel):
    user_id: str
    text: str
    top_k: int = 3  # Number of top results to fetch
    threshold: float = 0.75  # Threshold for similarity


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def retrieve_top_k_results(query_embedding, top_k, threshold):
    # Perform similarity search with FAISS
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    
    top_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:
            top_results.append(documents[idx])
    
    return top_results

# Function to cache the result in Redis
def cache_result(user_id, query_text, result, ttl=600):
    
    cache_key = f"{user_id}:{query_text}"
    redis_client.setex(cache_key, ttl, json.dumps(result))

# Function to check for cached results
def get_cached_result(user_id, query_text):
    cache_key = f"{user_id}:{query_text}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    return None

# API endpoint to search documents and generate context with caching
@app.post("/search")
async def search_documents(request: SearchRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    
    # Fetch user from database or create a new entry
    user = db.query(UserRequest).filter(UserRequest.user_id == user_id).first()
    
    if user:
        user.request_count += 1
    else:
        user = UserRequest(user_id=user_id, request_count=1)
        db.add(user)
    
    db.commit()
    
    # If user has exceeded the request limit, throw 429
    if user.request_count > 5:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    # Check for cached results
    cached_result = get_cached_result(user_id, request.text)
    if cached_result:
        return {
            "context": cached_result["context"],
            "documents": cached_result["documents"],
            "inference_time": "cached",
            "request_count": user.request_count
        }

    # Start measuring inference time
    start_time = time.time()
    
    # Encode the query text to an embedding
    query_embedding = model.encode(request.text, convert_to_tensor=False)
    
    # Retrieve top-k similar documents
    top_k_docs = retrieve_top_k_results(query_embedding, request.top_k, request.threshold)
    
    # If no documents found, raise an error
    if not top_k_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Aggregate context (concatenate the retrieved document texts)
    context = " ".join(top_k_docs)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Cache the result
    cache_result(user_id, request.text, {
        "context": context,
        "documents": top_k_docs
    })

    # Log the request details
    logging.info(f"User {user_id} made a request. Inference time: {inference_time:.2f} seconds. Request count: {user.request_count}")
    
    # Return the generated context along with metadata
    return {
        "context": context,
        "documents": top_k_docs,
        "inference_time": f"{inference_time:.2f} seconds",
        "request_count": user.request_count
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"message": "API is active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
