# 21BBS0119_ML
Legal Force [Trademarkia] 

1. User submits a query
2. Backend retrieves relevant documents based on embeddings 
3. Context is aggregated 
4. Pass the context to the LLM for generating the final response →


# Requirements:
1. FastAPI for building the API.
2. FAISS for efficient similarity search.
3. Sentence Transformers to create embeddings for the documents and queries.
4. PostgreSQL

Context generated should be concise as context length for LLMs can be less. Every LLM has its own context window length. 
![image](https://github.com/user-attachments/assets/1dae3cf2-8119-4a8f-978a-7d7b984808c3)
