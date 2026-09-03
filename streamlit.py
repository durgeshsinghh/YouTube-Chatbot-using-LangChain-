import streamlit as st
import requests


# -------------------------
# CONFIG
# -------------------------

st.set_page_config(
    page_title="YouTube Chatbot"
)

st.title("🎥 YouTube Transcript Chatbot")

st.write(
    "🚀 App Running Successfully"
)


# -------------------------
# FASTAPI URL
# -------------------------

API_URL = "http://127.0.0.1:8000"


# -------------------------
# VIDEO URL
# -------------------------

url = st.text_input(
    "Enter YouTube Video URL"
)


# -------------------------
# PROCESS VIDEO
# -------------------------

if st.button("Get Transcript"):

    if not url:

        st.warning(
            "Please enter a URL"
        )

    else:

        try:

            response = requests.post(
                f"{API_URL}/process-video",
                json={
                    "url": url
                }
            )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    data["message"]
                )

                st.session_state.ready = True

                st.info(
                    f"Video ID: {data['video_id']}"
                )

                st.info(
                    f"Chunks created: {data['chunks']}"
                )

            else:

                st.error(
                    response.json()["detail"]
                )

        except Exception as e:

            st.error(
                f"Backend error: {str(e)}"
            )


# -------------------------
# QUESTION
# -------------------------

if st.session_state.get(
    "ready",
    False
):

    query = st.text_input(
        "Ask a question about the video"
    )

    if st.button("Ask"):

        if not query:

            st.warning(
                "Please enter a question"
            )

        else:

            try:

                response = requests.post(

                    f"{API_URL}/ask",

                    json={
                        "question": query
                    }
                )

                if response.status_code == 200:

                    data = response.json()

                    st.write(
                        "### Answer:"
                    )

                    st.write(
                        data["answer"]
                    )

                else:

                    st.error(
                        response.json()["detail"]
                    )

            except Exception as e:

                st.error(
                    f"Backend error: {str(e)}"
                )
