--->Document Review System (RAG-Based)

This project is an AI-powered document review system that helps users understand large documents quickly. Users can upload a PDF and ask questions about the content, and the system retrieves relevant sections from the document before generating an answer. The application is built using a Retrieval-Augmented Generation (RAG) pipeline so the responses are grounded in the uploaded document rather than generic model knowledge.

-->What the System Does?

Upload a PDF document

Ask questions about the document

Retrieve the most relevant text sections

Generate answers using an AI model

Provide summaries or explanations of document content

The goal is to make reviewing long documents faster and easier.

-->How It Works?

The document is uploaded and converted into text.

The text is divided into smaller chunks.

Each chunk is converted into embeddings.

These embeddings are stored in a vector database.

When a user asks a question:

The system finds the most relevant chunks.

Those chunks are sent to the language model.

The model generates an answer using that context.

This process is known as Retrieval-Augmented Generation (RAG).

-->Technologies Used!

Python

Streamlit

LangChain

Hugging Face Transformers

Sentence Transformers

FAISS (Vector Search)

PyPDF2

-->Running the Model

This project can run in two ways:

1] Local Model

The language model can run locally using Hugging Face Transformers if the required models are downloaded.

2] Hugging Face API

The system can also call the Hugging Face inference API.
Users can enter their own Hugging Face API token to run the model without downloading it locally. This helps reduce memory usage and makes deployment easier.
