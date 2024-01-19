import streamlit as st
import torch
import whisper
import pytube
import librosa
import IPython.display as ipd

# Load the Whisper model
model_m = whisper.load_model('medium')

def transcribe_audio(file_path):
    try:
        transcription = model_m.transcribe(file_path, fp16=False)['text']
        return transcription
    except Exception as e:
        return str(e)

def main():
    st.title("Audio Transcription App")

    uploaded_file = st.file_uploader("Upload an MP3 audio file", type=["mp3"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        file_path = "temp_audio_file.mp3"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.audio(file_path, format="audio/mp3", start_time=0)

        if st.button("Transcribe"):
            transcription = transcribe_audio(file_path)
            st.write("Transcription:")
            st.write(transcription)

    st.markdown(
        "Note: This app uses the Whisper ASR model for transcription. It may not be perfect and accuracy may vary."
    )

if __name__ == "__main__":
    main()
