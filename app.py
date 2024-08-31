import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emotion to emoji mapping
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮" , "love":"❤️"
}

# Predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Get prediction probability
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Main function
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if raw_text.strip():  # Ensure input is not empty
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', 
                    y='probability', 
                    color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)
        else:
            st.error("Please enter some text to analyze.")

if __name__ == '__main__':
    main()
