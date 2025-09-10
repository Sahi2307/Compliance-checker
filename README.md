
## Compliance Checker - RAG Application with Gemini AI

A comprehensive compliance checking application that uses Google Gemini AI for document analysis and PostgreSQL with pgvector for vector storage and similarity search.

### Key Features

* **Document Analysis**: Upload and analyze compliance documents using Gemini AI
* **Vector Search**: Store document embeddings in PostgreSQL with pgvector extension
* **Similarity Search**: Find similar documents and sections using cosine similarity
* **Batch Processing**: Process multiple documents simultaneously
* **Report Generation**: Generate detailed compliance analysis reports

## Why PostgreSQL + pgvector?

Using PostgreSQL with pgvector as your vector database offers several key advantages over dedicated vector databases:

* PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.
* By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.
* pgvector provides efficient vector similarity search with support for various distance metrics including cosine similarity, L2 distance, and inner product.
* TimescaleDB integration offers time-series capabilities for tracking document processing and analysis over time.

## Prerequisites

* Docker
* Python 3.7+
* Google Gemini API key
* PostgreSQL GUI client (TablePlus recommended)

## Quick Start

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client (TablePlus recommended)
3. Configure environment variables with your Gemini API key
4. Install Python dependencies
5. Run the application

## Detailed Instructions

### 1. Set up Docker environment

Build and start the TimescaleDB container:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

The database will be available at:
* Host: localhost
* Port: 5433
* User: postgres
* Password: password
* Database: postgres

### 2. Connect to the database using a PostgreSQL GUI client

* Open TablePlus (or your preferred PostgreSQL client)
* Create a new connection with the following details:  
   * Host: localhost  
   * Port: 5433  
   * User: postgres  
   * Password: password  
   * Database: postgres
   * Schema: public

### 3. Configure environment variables

Create a `.env` file in the root directory:

```bash
TIMESCALE_SERVICE_URL=postgresql://postgres:password@localhost:5433/postgres
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Install Python dependencies

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 5. Initialize the database

Create the necessary database extensions and tables:

```bash
docker exec -it timescaledb psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb; CREATE EXTENSION IF NOT EXISTS pgcrypto; CREATE EXTENSION IF NOT EXISTS vector;"
```

### 6. Load data into vector table

```bash
python app\insert_vectors.py
```

### 7. Run the Streamlit applications

```bash
# Single PDF Q&A
python -m streamlit run app\main.py

# Multiple PDFs (batch + Q&A per PDF)
python -m streamlit run app\multiple.py
```

## Architecture

* **Vector Database**: PostgreSQL (TimescaleDB) + `pgvector` extension
* **Table**: `public.embedding_1`
  - Columns: `id UUID`, `metadata JSONB`, `contents TEXT`, `embedding VECTOR(768)`, `created_at TIMESTAMPTZ`
  - Index: HNSW on `embedding` column for fast similarity search
* **Embeddings**: Google Gemini `text-embedding-004` (768 dimensions)
* **LLM**: Gemini (`gemini-1.5-pro` or `gemini-1.5-flash`)
* **Frontend**: Streamlit web interface

## Usage

### Single Document Analysis
1. Run the main application: `python -m streamlit run app\main.py`
2. Upload a PDF document
3. Ask questions about the document
4. View similarity search results

### Batch Document Processing
1. Run the multiple documents app: `python -m streamlit run app\multiple.py`
2. Upload multiple PDF documents
3. Process all documents in batch
4. Generate individual analysis reports

## Database Schema

### Create Table and Index

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.embedding_1 (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  metadata JSONB,
  contents TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS embedding_1_embedding_hnsw_idx
ON public.embedding_1 USING hnsw (embedding vector_cosine_ops);
```

### Vector Search Optimization

The application uses HNSW (Hierarchical Navigable Small World) indexing for fast similarity search:

* **HNSW Index**: Provides fast approximate nearest neighbor search
* **Cosine Similarity**: Uses `vector_cosine_ops` for semantic similarity
* **768 Dimensions**: Compatible with Gemini's `text-embedding-004` model

## Troubleshooting

### Common Issues

* **TablePlus shows nothing**: Ensure you're connected to port `5433` and database `postgres`, schema `public`
* **`type "vector" does not exist`**: Run `CREATE EXTENSION vector;` in the target database
* **Embedding dimension mismatch**: Ensure database column is `VECTOR(768)` and matches Gemini's embedding dimensions
* **Gemini API errors**: Check your API key and quota limits
* **Connection refused**: Ensure Docker container is running with `docker ps`

### Reports

* Generated PDF analysis reports are saved under `reports/` directory
* Reports include document analysis, compliance findings, and recommendations
* Each report is timestamped and can be downloaded from the Streamlit interface

## References

* [pgvector GitHub Repository](https://github.com/pgvector/pgvector)
* [Google Gemini API Documentation](https://ai.google.dev/docs)
* [TimescaleDB Documentation](https://docs.timescale.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is for educational and research purposes. Please ensure compliance with Google's Gemini API terms of service and any applicable data protection regulations.
