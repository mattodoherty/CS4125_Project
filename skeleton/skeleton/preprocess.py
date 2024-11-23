#Methods related to data loading and all pre-processing steps will go here

import pandas as pd
import re
import logging
from googletrans import Translator
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import classification_report


def preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Initial DataFrame shape: {df.shape}")

    # Ensure data type compatibility and strip whitespace
    df['Interaction content'] = df['Interaction content'].astype(str).str.strip()
    df['Ticket Summary'] = df['Ticket Summary'].astype(str).str.strip()

    # Rename columns for easier processing
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df['Interaction content']
    df["y"] = df["y2"]

    # Remove rows with empty or NaN target labels
    df = df.loc[(df["y"] != '') & (~df["y"].isna())]
    print(f"After removing empty target labels: {df.shape}")

    # Deduplication and noise removal
    df = de_duplication(df)
    df = noise_remover(df)

    # Translation of relevant columns with fallback
    print("Translating 'Ticket Summary'...")
    df["Ticket Summary"] = batch_translate_to_en(df["Ticket Summary"].tolist())
    print("Translating 'Interaction content'...")
    df["Interaction content"] = batch_translate_to_en(df["Interaction content"].tolist())

    # Remove rows with empty translations in 'Ticket Summary'
    df = df[df["Ticket Summary"].str.strip().astype(bool)]

    return df






def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT], keep='first')


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    def clean_text(text: str) -> str:

        if not isinstance(text, str):  # Handle invalid inputs gracefully
            return "Invalid content"

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but retain meaningful punctuation
        text = re.sub(r'[^A-Za-z0-9.,?!\s]', '', text)

        return text.strip()

    # Apply cleaning to the required columns
    if Config.TICKET_SUMMARY in df.columns:
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)

    if Config.INTERACTION_CONTENT in df.columns:
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)

    return df


def batch_translate_to_en(texts):
    from googletrans import Translator

    translator = Translator()
    translated_texts = []

    for i in range(0, len(texts), 10):  # Batch size of 10
        batch = [
            text if isinstance(text, str) and text.strip() else "No content available"
            for text in texts[i:i + 10]
        ]
        try:
            translations = translator.translate(batch, src="auto", dest="en")
            translated_texts.extend(
                [
                    t.text if t and hasattr(t, "text") else batch[idx]
                    for idx, t in enumerate(translations)
                ]
            )
        except Exception as e:
            print(f"Translation batch failed: {e}")
            # Fallback: retain original text for failed translations
            translated_texts.extend(batch)

    # Ensure output length matches input
    return translated_texts

def get_input_data(file_path: str = "data_uef/AppGallery_done.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Convert the dtype object to Unicode string (if necessary)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Optional: Rename variables for easier processing
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df[Config.INTERACTION_CONTENT]
    df["y"] = df["y2"]

    # Remove rows where "y" is empty or NaN
    df = df.loc[(df["y"] != '') & (~df["y"].isna()), :]
    return df


def get_embeddings(df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_converter = TfidfVectorizer(max_features=2000, min_df=0.001, max_df=0.95)

    # Handle Ticket Summary
    df["Ticket Summary"] = df["Ticket Summary"].fillna("missing")
    if df["Ticket Summary"].dropna().str.strip().empty:
        print("Warning: Ticket Summary is empty. Using placeholder embeddings.")
        ticket_summary_embeddings = np.zeros((len(df), 1))
    else:
        try:
            ticket_summary_embeddings = tfidf_converter.fit_transform(df["Ticket Summary"]).toarray()
        except ValueError as e:
            print(f"Error with TfidfVectorizer: {e}")
            ticket_summary_embeddings = np.zeros((len(df), 1))

    # Handle Interaction Content
    df["Interaction content"] = df["Interaction content"].fillna("missing")
    if df["Interaction content"].dropna().str.strip().empty:
        print("Warning: Interaction content is empty. Using placeholder embeddings.")
        interaction_embeddings = np.zeros((len(df), 1))
    else:
        try:
            interaction_embeddings = tfidf_converter.fit_transform(df["Interaction content"]).toarray()
        except ValueError as e:
            print(f"Error with TfidfVectorizer: {e}")
            interaction_embeddings = np.zeros((len(df), 1))

    embeddings = np.concatenate((interaction_embeddings, ticket_summary_embeddings), axis=1)
    df["Embeddings"] = embeddings.tolist()  # Add embeddings as a column for reference
    return embeddings, df




