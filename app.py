import gradio as gr
import whisper
import os
from groq import Groq

# 🔐 Get Groq API key securely from Hugging Face Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama3-8b-8192"

# 🎙 Load Whisper
transcriber = whisper.load_model("base")

def transcribe_and_summarize(audio):
    # Step 1: Transcribe + Detect Language
    result = transcriber.transcribe(audio)
    transcript = result["text"]
    detected_lang = result["language"]

    # Step 2: Summarize in the same language
    if detected_lang == "en":
        system_prompt = "You are an expert English summarizer."
        user_prompt = f"Please summarize the following English text:\n\n{transcript}"
    elif detected_lang == "ur":
        system_prompt = "آپ ایک ماہر خلاصہ نگار ہیں جو اردو میں خلاصہ فراہم کرتے ہیں۔"
        user_prompt = f"براہ کرم مندرجہ ذیل اردو متن کا خلاصہ فراہم کریں:\n\n{transcript}"
    else:
        system_prompt = "You are a helpful summarizer."
        user_prompt = f"Summarize this text:\n\n{transcript}"

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    summary = response.choices[0].message.content.strip()

    lang_label = "English" if detected_lang == "en" else "Urdu" if detected_lang == "ur" else detected_lang.upper()

    return f"[{lang_label}] {transcript}", f"[{lang_label}] {summary}"

demo = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=gr.Audio(type="filepath", label="🎧 Upload Audio (English or Urdu)"),
    outputs=[
        gr.Textbox(label="📝 Transcript"),
        gr.Textbox(label="🧠 Summary")
    ],
    title="🗣️ Multilingual Audio Summarizer",
    description="Upload English or Urdu audio. The app transcribes and summarizes in the same language using Whisper + Groq."
)

demo.launch()
