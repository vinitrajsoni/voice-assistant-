import streamlit as st
from st_audiorec import st_audiorec
from config import AUDIO_FILE
from audio_utils import save_audio_from_browser, text_to_speech
from bulbul_voice import transcribe_with_sarvam
from llm_chain import load_qa_chain

st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")
st.title("🎙️ Gemini RAG Chatbot with Voice & Text")
st.markdown("Ask anything from your documents - via Voice or Text!")

qa_chain = load_qa_chain()

# --- Record Voice via Browser ---
audio_bytes = st_audiorec()
if audio_bytes:
    save_audio_from_browser(audio_bytes, AUDIO_FILE)
    st.audio(AUDIO_FILE, format="audio/wav")

    with st.spinner("🧠 Transcribing with Sarvam AI..."):
        transcript, detected_lang_code = transcribe_with_sarvam(AUDIO_FILE)

        if not transcript:
            st.error("❌ Failed to transcribe.")
        else:
            st.markdown(f"**📝 Transcript:** `{transcript}`")
            st.markdown(f"**🌐 Detected Language Code:** `{detected_lang_code}`")

            with st.spinner("🤖 Generating AI response..."):
                result = qa_chain.invoke(transcript, detected_lang_code)
                reply_text = result["result"].content

                st.markdown(reply_text)

                st.success("📜 Answer:")
                # st.markdown(reply_text)


            with st.spinner("🎧 Converting to speech..."):
                audio_base64 = text_to_speech(reply_text, detected_lang_code)
                if audio_base64:
                    st.markdown("### 🔈 Voice Output")
                    st.markdown(
                        f"""
                        <audio autoplay controls>
                            <source src=\"data:audio/wav;base64,{audio_base64}\" type=\"audio/wav\">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )

# --- Text Input ---
st.markdown("---")
query = st.text_input("💬 Enter your text question:")
if query:
    with st.spinner("Generating response..."):
        result = qa_chain.invoke(query, "en-IN")
        reply_text = result.get("result", "")
        st.success("📜 Answer:")
        st.write(reply_text)