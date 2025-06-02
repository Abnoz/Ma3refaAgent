# Ma3refa Agent

A Python-based document QA system that processes PDFs from Azure Blob Storage and enables Arabic language queries using Azure OpenAI and Azure Cognitive Search.

## Features

- Downloads PDFs from Azure Blob Storage
- Extracts text using LangChain's PyPDFLoader
- Generates embeddings using Azure OpenAI
- Stores vectors in Azure Cognitive Search
- Provides a CLI interface for Arabic language queries
- Uses GPT-4 for generating answers

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Azure credentials:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` with your Azure service credentials

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Download PDFs from your Azure Blob Storage container
2. Process and index the documents
3. Start an interactive CLI where you can ask questions in Arabic

## Environment Variables

Required environment variables in `.env`:

- `AZURE_STORAGE_CONNECTION_STRING`: Azure Blob Storage connection string
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Name of embedding model deployment
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Name of chat model deployment
- `AZURE_SEARCH_ENDPOINT`: Azure Cognitive Search endpoint
- `AZURE_SEARCH_KEY`: Azure Cognitive Search admin key
- `AZURE_SEARCH_INDEX_NAME`: Name for the search index
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version # Ma3refaAgent
