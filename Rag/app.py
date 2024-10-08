import streamlit as st
import nest_asyncio
import logging
import os
from typing import List
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

load_dotenv()
os.getenv("OPENAI_API_KEY")

