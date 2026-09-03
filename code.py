import os
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)


# -------------------------
# CONFIG
# -------------------------

load_dotenv()

app = FastAPI(
    title="YouTube Chatbot API"
)


# -------------------------
# GEMINI MODEL
# -------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# -------------------------
# REQUEST MODELS
# -------------------------

class VideoRequest(BaseModel):
    url: str


class QuestionRequest(BaseModel):
    question: str


# -------------------------
# GLOBAL VECTORSTORE
# -------------------------

vectorstore = None


# -------------------------
# GET VIDEO ID
# -------------------------

def get_video_id(url):

    parsed_url = urlparse(url)

    if parsed_url.hostname in [
        "www.youtube.com",
        "youtube.com"
    ]:

        return parse_qs(
            parsed_url.query
        ).get("v", [None])[0]

    elif parsed_url.hostname == "youtu.be":

        return parsed_url.path[1:]

    return None


# -------------------------
# FORMAT DOCUMENTS
# -------------------------

def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/")
def home():

    return {
        "message": "YouTube Chatbot API is running"
    }


# -------------------------
# PROCESS VIDEO
# -------------------------

@app.post("/process-video")
def process_video(request: VideoRequest):

    global vectorstore

    video_id = get_video_id(request.url)

    if not video_id:

        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL"
        )

    try:

        # Get transcript

        api = YouTubeTranscriptApi()

        transcript_list = api.list(video_id)

        transcript = transcript_list.find_transcript(
            [
                t.language_code
                for t in transcript_list
            ]
        )

        data = transcript.fetch()

        full_text = " ".join(
            [
                entry.text
                for entry in data
            ]
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Transcript blocked or unavailable"
        )

    # -------------------------
    # SPLIT TRANSCRIPT
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(
        full_text
    )

    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="Transcript is empty"
        )

    # -------------------------
    # EMBEDDINGS
    # -------------------------

    embeddings = FakeEmbeddings(
        size=384
    )

    # -------------------------
    # FAISS
    # -------------------------

    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )

    return {
        "message": "Transcript processed successfully",
        "video_id": video_id,
        "chunks": len(chunks)
    }


# -------------------------
# ASK QUESTION
# -------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    global vectorstore

    if vectorstore is None:

        raise HTTPException(
            status_code=400,
            detail="Please process a video first"
        )

    try:

        # -------------------------
        # RETRIEVER
        # -------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

        # -------------------------
        # RAG CHAIN
        # -------------------------

        parallel_chain = RunnableParallel({

            "context":
                retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough()
        })

        # -------------------------
        # PROMPT
        # -------------------------

        prompt = PromptTemplate.from_template(
            """
Answer the question based only
on the context below.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # -------------------------
        # COMPLETE CHAIN
        # -------------------------

        main_chain = (
            parallel_chain
            | prompt
            | model
            | StrOutputParser()
        )

        # -------------------------
        # GENERATE ANSWER
        # -------------------------

        result = main_chain.invoke(
            request.question
        )

        return {
            "answer": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
