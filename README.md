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

<img width="517" alt="image" src="https://github.com/user-attachments/assets/ac8b1a0f-1b29-4e9b-8ac5-350c610d2ad6">
Reference of image - https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html

Context generated should be concise as context length for LLMs can be less. Every LLM has its own context window length. 
![image](https://github.com/user-attachments/assets/1dae3cf2-8119-4a8f-978a-7d7b984808c3)

<img width="670" alt="image" src="https://github.com/user-attachments/assets/e080f52f-c9e0-4f03-9bcd-487823d81e31">

👆Response body gives context, top 3 documents, inference time and number of requests from the respective user.

<img width="670" alt="image" src="https://github.com/user-attachments/assets/d8adaba2-a0bd-4b80-9d74-6b8456b1ab7a">

👆 After number of requests exceeded 



This Medium Article helped me to download and run redis on windows -> https://naveenrenji.medium.com/install-redis-on-windows-b80880dc2a36 ✨
