import streamlit as st
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# -----------------------------
# CONFIG + UI (ALWAYS LOAD FAST)
# -----------------------------
st.set_page_config(page_title="YouTube Chatbot")

st.title("🎥 YouTube Transcript Chatbot")
st.write("🚀 App Running Successfully")
st.write("PORT:", os.environ.get("PORT"))

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# MODEL
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------
# HELPERS
# -----------------------------
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_video_id(url):
    parsed_url = urlparse(url)
    
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    
    return None 

# -----------------------------
# INPUT
# -----------------------------
url = st.text_input("Enter YouTube Video URL")

# -----------------------------
# PROCESS VIDEO (LAZY LOADING HERE)
# -----------------------------
if st.button("Get Transcript"):

    if not url:
        st.warning("Please enter a URL")

    else:
        video_id = get_video_id(url)

        if video_id:
            try:
                st.info("📥 Fetching transcript...")

                # ✅ NEW API
                api = YouTubeTranscriptApi()
                transcript_list = api.list(video_id)

                transcript = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                )

                data = transcript.fetch()
                full_text = " ".join([entry.text for entry in data])

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_text(full_text)

                st.info("🧠 Loading embeddings (first time only)...")

                # 🔥 LAZY LOAD EMBEDDINGS HERE
                from langchain_huggingface import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.info("📦 Creating vector store...")

                vectorstore = FAISS.from_texts(chunks, embeddings)

                # Store
                st.session_state.vectorstore = vectorstore
                st.session_state.ready = True

                st.success("✅ Transcript processed successfully!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        else:
            st.error("Invalid YouTube URL")

# -----------------------------
# QUERY SECTION
# -----------------------------
if st.session_state.get("ready"):

    query = st.text_input("Ask a question about the video")

    if query:
        try:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            prompt = PromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

            main_chain = parallel_chain | prompt | model | StrOutputParser()

            result = main_chain.invoke(query)

            st.write("### 💬 Answer:")
            st.write(result)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")