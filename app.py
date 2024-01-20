import streamlit as st
import whisper
import time

header = st.container()
explain = st.container()
pre = st.container()

with header:

    st.title("Convert Speech to Text")
    st.subheader("Note: This Model can Only Trascribe from English Speech to English Text")
    st.write("1. Upload Audio File \n 2. Wait till Transcribe Button Appears \n 3. Click on transcribe button and wait till result generates \n 4. Can  download the text in .txt format \n 5. If you Liked it then Press the Button.")

with explain: 
    # Upload audio file with streamlit
    audio_file = st.file_uploader("Upload Audio", type=["wav","mp3","m4a"])

    model = whisper.load_model("medium")

    def get_audio_file_details(file):
        file_details = {"FileName": file.name, "FileType": file.type, "FileSize":file.size}
        return file_details

    if st.sidebar.button("Transcribe Audio"):
        if audio_file is not None:
            st.sidebar.success("Transcribing Audio")
            transcription = model.transcribe(audio_file.name)
            st.sidebar.success("Transcription Complete")
            result = transcription["text"]
            st.markdown(result)
            st.download_button(label = 'Download Text',data= result,file_name = "Result.txt")
        else:
            st.sidebar.error("Please Upload an Audio File")

    st.sidebar.header("Play Original Audio File")
    st.sidebar.audio(audio_file)

with pre:
    if st.sidebar.button("Press if you Like It"):
        st.balloons()
