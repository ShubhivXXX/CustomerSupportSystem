from dotenv import load_dotenv
load_dotenv()

import streamlit as st
# import google.generativeai as genai
import os
from google import genai

client = genai.Client(api_key=os.getenv("google_api_key"))


def get_gemini_response(question):
    response = client.models.generate_content(model="gemini-flash-latest",contents=question)
    return response.text

st.set_page_config(page_title="FirstQ&A Demo")
st.header("GeminiLLM App")
input = st.text_input("input: ",key="input")
submit = st.button("Ask")

if submit:
    response = get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)






# from dotenv import load_dotenv
# load_dotenv()

# import streamlit as st
# import os
# from google import genai

# # Initialize the client with your API key
# client = genai.Client(api_key=os.getenv("google_api_key"))

# def get_gemini_response(question):
#     response = client.models.generate_content(
#         model="gemini-flash-latest",
#         contents=question
#     )
#     return response.text  # correct attribute

# st.set_page_config(page_title="First Q&A Demo")
# st.header("GeminiLLM App")

# user_input = st.text_input("input: ", key="input")
# if st.button("Ask"):
#     answer = get_gemini_response(user_input)
#     st.subheader("The Response is")
#     st.write(answer)

