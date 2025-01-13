import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer
tfidf = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Configure Streamlit page
st.set_page_config(page_title="Text Analysis App", layout="wide")

# Sidebar navigation
menu = st.sidebar.radio("Menu", ["Pre-Processing & Data Tables", "Single Sentence Analysis"])

# Initialize session state for data storage
if "data" not in st.session_state:
    st.session_state["data"] = None

if menu == "Pre-Processing & Data Tables":
    st.header("Pre-Processing & Data Tables")

    # Upload data file and stopwords
    data_file = st.file_uploader("Upload Data File (CSV/TXT)", type=['csv', 'txt'])
    stopword_file = st.file_uploader("Upload Stopword File (TXT)", type=['txt'])
    remove_stopwords = st.checkbox("Remove Stopwords")

    if st.button("Submit"):
        if data_file:
            # Load data
            if data_file.name.endswith('.csv'):
                data = pd.read_csv(data_file, header=None, names=["Sentence"])
            else:
                data = pd.read_csv(data_file, sep="\n", header=None, names=["Sentence"])

            # Process stopwords if uploaded
            if stopword_file and remove_stopwords:
                stopwords = stopword_file.read().decode('utf-8').splitlines()
                data["Processed"] = data["Sentence"].apply(
                    lambda x: " ".join(
                        [word for word in x.split() if word.lower() not in stopwords]
                    )
                )
            else:
                data["Processed"] = data["Sentence"]

            # Save data to session state
            st.session_state["data"] = data
            st.success("Data successfully loaded!")
        else:
            st.error("Please upload a data file first.")

    if st.session_state["data"] is not None:
        data = st.session_state["data"]

        # Display pre-processed data
        st.subheader("Pre-Processed Data")
        st.dataframe(data)

        # Predict sentiment if data is available
        X = tfidf.transform(data["Processed"])
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Add analysis results to the data
        data["Positive"] = probabilities[:, 1]
        data["Negative"] = probabilities[:, 0]
        data["Analysis"] = ["Positive" if p == 1 else "Negative" for p in predictions]

        # Display analysis table
        st.subheader("Data Tables With Full Features")
        st.dataframe(data[["Sentence", "Processed", "Positive", "Negative", "Analysis"]])

        # Sentiment distribution visualization
        st.subheader("Sentiment Analysis Visualization")
        sentiment_counts = data["Analysis"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="viridis")
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

elif menu == "Single Sentence Analysis":
    st.header("Single Sentence Analysis")

    # User input for a single sentence
    sentence = st.text_input("Enter a sentence for analysis:", placeholder="Example: This cake tastes amazing")

    if st.button("Analyze Sentence"):
        if sentence:
            # Simple preprocessing (add more steps if needed)
            processed_sentence = " ".join(
                [word for word in sentence.lower().split()]
            )

            # Transform and predict
            X = tfidf.transform([processed_sentence])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            # Display results
            st.write(f"**Sentence:** {sentence}")
            st.write(f"**Positive Probability:** {probabilities[1]:.4f}")
            st.write(f"**Negative Probability:** {probabilities[0]:.4f}")
            st.write(f"**Analysis Result:** {'Positive' if prediction == 1 else 'Negative'}")
        else:
            st.error("Please enter a sentence first.")
