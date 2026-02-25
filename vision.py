from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from google import genai
from PIL import Image
from google.genai import types
client = genai.Client(api_key=os.getenv("google_api_key"))


def get_gemini_response(question,uploaded_file):
    imagePart= types.Part.from_bytes(data = uploaded_file.getvalue(), mime_type=uploaded_file.type)
    questionPart= types.Part.from_text(text=question)
    contents = [
        imagePart,  questionPart
    ]
    response = client.models.generate_content(model="gemini-flash-latest",contents=contents)
    return response.text

st.set_page_config(page_title="FirstQ&A Demo")
st.header("GeminiLLM App")
input = st.text_input("input: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


submit = st.button("Ask")

if submit:
    response = get_gemini_response(input,uploaded_file)
    st.subheader("The Response is")
    st.write(response)



