import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import queue
import numpy as np
from scipy.io.wavfile import write
from config import AUDIO_FILE
from audio_utils import save_audio_from_browser, text_to_speech
from bulbul_voice import transcribe_with_sarvam
from llm_chain import load_qa_chain

st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")
st.title("🎙️ Gemini RAG Chatbot with Voice & Text")
st.markdown("Ask anything from your documents - via Voice or Text!")

qa_chain = load_qa_chain()

# --- Setup queue to collect audio frames ---
audio_queue = queue.Queue()

# --- Audio callback ---
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio_queue.put(frame.to_ndarray().flatten())
    return frame

# --- Stream component ---
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    audio_frame_callback=audio_frame_callback,
    client_settings=ClientSettings(
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

# --- Save audio once recording stops ---
if webrtc_ctx.state.playing == False and not audio_queue.empty():
    samples = np.concatenate(list(audio_queue.queue)).astype(np.int16)
    write(AUDIO_FILE, 48000, samples)
    st.success("🔊 Audio saved as input.wav")
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

                st.success("📜 Answer:")
                st.markdown(reply_text)

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

