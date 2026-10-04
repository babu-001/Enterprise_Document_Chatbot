# PDF Q&A: React + FastAPI + LangChain + OpenAI + Qdrant (Docker)

Flow: Upload PDF -> PyPDFLoader extracts text -> RecursiveCharacterTextSplitter (1000/150) -> OpenAI embeddings -> Qdrant -> similarity search -> LangChain chain (prompt | ChatOpenAI | parser) -> answer with [n] citations.

## Deploy with Docker
    cp .env.example .env        # put your OPENAI_API_KEY
    docker compose up --build -d
- App: http://localhost:3000
- API docs: http://localhost:8000/docs
- Stop: `docker compose down` (data persists in the `qdrant_storage` volume)

If you change OPENAI_EMBED_MODEL/DIM, delete the volume (`docker compose down -v`) so the collection is recreated.

## Local dev (without Docker)
    docker run -p 6333:6333 qdrant/qdrant
    cd backend && pip install -r requirements.txt && QDRANT_URL=http://localhost:6333 uvicorn app.main:app --reload
    cd frontend && npm install && npm run dev     # http://localhost:5173
