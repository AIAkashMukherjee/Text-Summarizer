import streamlit as st
import os
from src.pipeline.prediction_pipeline import PredictionPipeline

# Streamlit app title
st.title("Text Summarization App")

# Text input for summarization
text_input = st.text_area("Enter text for summarization:", "What is Text Summarization?")

# Button to trigger prediction
if st.button("Summarize"):
    if text_input:
        try:
            obj = PredictionPipeline()
            summary = obj.predict(text_input)
            st.success("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error Occurred! {e}")
    else:
        st.warning("Please enter some text to summarize.")

# Button to trigger training
# if st.button("Train Model"):
#     try:
#         os.system("python main.py")
#         st.success("Training successful!")
#     except Exception as e:
#         st.error(f"Error Occurred! {e}")
