# News Demo

- RAG Source: BBC News Archive (http://mlg.ucd.ie/datasets/bbc.html)
- Web Search powered by Tavily
- Local FAISS Vector Store

App can answer questions on current news sources, and the BBC Archive from 2004-2005.

### Run Instructions

Package Dependencies
- Build from toml file and uv

Create the vector DB
- Download and unzip to /data folder within root. Should yield 'bbc' folder with nested sub-category folders
- Run the 'loadarchive.ipynb' notebook to create and load vector embeddings
- Smoke-test with a few queries

Run the app
- Create a .env file with your API keys
- Run newschat.py
- Open browser to run Gradio
