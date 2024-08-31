Your code looks mostly correct, but based on the error message and the provided code, here are a few potential issues to consider:

### 1. **KeyError in Dictionary Lookup**:
   The error you encountered seems to be related to the dictionary lookup in `emotions_emoji_dict`. If the `prediction` key is not found in `emotions_emoji_dict`, a `KeyError` will occur.

   **Solution**: You can add a safeguard using the `get` method, which allows you to specify a default value if the key is not found.

   ```python
   emoji_icon = emotions_emoji_dict.get(prediction, "❓")
   ```

   This will return a default emoji (e.g., `"❓"`) if the `prediction` key is not found in the dictionary.

### 2. **Model Prediction**:
   Ensure that the model (`pipe_lr`) is correctly loaded and that it predicts classes that exist within `emotions_emoji_dict`. If the model predicts a class not included in the dictionary, it will cause an error.

   **Check the Model Output**:
   ```python
   prediction = predict_emotions(raw_text)
   st.write("Prediction:", prediction)
   ```

   This will help you verify what the model is actually predicting.

### 3. **Correct Mapping in the Dictionary**:
   Double-check that all possible predictions from your model are covered by the `emotions_emoji_dict`. For example, if your model predicts `"joy"` but your dictionary does not have this key, it will raise a `KeyError`.

### 4. **Handling Edge Cases**:
   In some cases, `predict_proba` might return an empty result if the input is invalid or the model encounters an unexpected situation.

   **Check for Empty Input**:
   Ensure that `raw_text` is not empty or invalid before running predictions:
   
   ```python
   if raw_text.strip():
       prediction = predict_emotions(raw_text)
       probability = get_prediction_proba(raw_text)
   else:
       st.error("Please enter some text to analyze.")
   ```

### 5. **Library and File Dependencies**:
   Make sure all the dependencies (`joblib`, `streamlit`, `altair`, `numpy`, `pandas`) are correctly installed and the `text_emotion.pkl` model file is present in the correct directory.

### Example Revised Code:

Here’s a slightly revised version of your code that incorporates the error-handling mechanisms:

```python
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
    "shame": "😳", "surprise": "😮"
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
```

### Summary:
- Use `get` for safe dictionary lookups.
- Ensure the model's predictions match the keys in the dictionary.
- Handle empty or invalid inputs gracefully.

If you implement these suggestions, the issue should be resolved, and your app should work as expected!
