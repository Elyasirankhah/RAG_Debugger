## Development
Contributions and improvements are welcome!
# RAG Debugger v0.1

A minimal RAG (Retrieval-Augmented Generation) debugger that shows how retrieved documents influence generated answers. This tool helps you understand which parts of your answer are supported by the source documents and which are not.

## Features

- **Document Upload**: Upload PDF or TXT files
- **Semantic Search**: Uses FAISS for efficient vector similarity search
- **Answer Generation**: Generates answers using OpenAI's Chat Completion API
- **Evidence Alignment**: Shows which chunks support each sentence in the answer
- **Unsupported Detection**: Flags sentences that have no supporting evidence

## Architecture

```
┌─────────────┐
│   FastAPI   │
│   Backend   │
└──────┬──────┘
       │
       ├─── Upload → Load → Chunk → Embed → Store (FAISS)
       │
       └─── Ask → Retrieve → Generate → Debug → Return
```

### Components

- **loader.py**: Loads PDF/TXT files
- **chunker.py**: Splits documents into overlapping chunks
- **embeddings.py**: Generates embeddings using OpenAI API
- **retriever.py**: FAISS-based vector store for similarity search
- **generator.py**: Generates answers using OpenAI Chat Completion
- **debugger.py**: Analyzes evidence alignment between answer sentences and chunks

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Start the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Upload Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

Response:
```json
{
  "document_ids": ["uuid1", "uuid2"]
}
```

#### 2. Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

Response:
```json
{
  "answer": "The main topic is...",
  "retrieved_chunks": [
    {
      "doc_id": "uuid1",
      "chunk_id": "uuid1_chunk_0",
      "text": "Chunk text...",
      "score": -0.123
    }
  ],
  "evidence_alignment": [
    {
      "answer_sentence": "The main topic is...",
      "supporting_chunks": ["uuid1_chunk_0"]
    }
  ],
  "unsupported_sentences": []
}
```

#### 3. Health Check

```bash
curl http://localhost:8000/health
```

## How It Works

### 1. Document Processing

When you upload documents:
1. Files are loaded (PDF or TXT)
2. Text is split into overlapping chunks (500 chars, 50 char overlap)
3. Each chunk is embedded using OpenAI's `text-embedding-ada-002`
4. Embeddings are stored in a FAISS index

### 2. Question Answering

When you ask a question:
1. Question is embedded using the same model
2. Top-k most similar chunks are retrieved using FAISS
3. Answer is generated using OpenAI Chat Completion with ONLY the retrieved chunks as context
4. Answer is split into sentences
5. Each sentence is compared with retrieved chunks using cosine similarity
6. Sentences with similarity < threshold (0.7) are flagged as unsupported

### 3. Evidence Alignment

The debugger:
- Splits the answer into sentences
- Computes semantic similarity between each sentence and each retrieved chunk
- If similarity ≥ threshold, the chunk is considered supporting evidence
- Sentences with no supporting chunks are flagged as "unsupported"

## Configuration

You can adjust these parameters in the code:

- **Chunk size**: Default 500 characters (in `app.py`)
- **Chunk overlap**: Default 50 characters (in `app.py`)
- **Top-k retrieval**: Default 5 chunks (in `app.py`)
- **Similarity threshold**: Default 0.7 (in `app.py`, debugger initialization)
- **LLM model**: Default `gpt-3.5-turbo` (in `rag/generator.py`)
- **Embedding model**: Default `text-embedding-ada-002` (in `rag/embeddings.py`)

## Limitations

- **In-memory storage**: Documents are stored in memory. Restarting the server clears all data.
- **Simple sentence splitting**: Uses regex-based sentence splitting. For production, consider using spaCy or NLTK.
- **No caching**: Chunk embeddings are regenerated during evidence analysis (could be optimized).
- **CPU-only**: Uses FAISS CPU version (no GPU acceleration).

## Web UI

A simple web interface is included in the `ui/` folder. Access it at `http://localhost:8000/ui` when the server is running.

Features:
- Drag-and-drop document upload
- Interactive question interface
- Visual evidence alignment
- Highlighted unsupported sentences
- Retrieved chunks with scores

## Future Improvements

- Add persistent storage (database)
- Implement embedding caching
- Add more sophisticated sentence splitting
- Support more file formats
- Add batch processing for multiple questions
- Improve UI with better visualization

## License

This is a minimal implementation for debugging and educational purposes.
