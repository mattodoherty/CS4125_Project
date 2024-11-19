from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def get_tfidf_embeddings(df: pd.DataFrame) -> np.ndarray:
    tfidf_converter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidf_converter.fit_transform(df["Interaction content"]).toarray()
    x2 = tfidf_converter.fit_transform(df["Ticket Summary"]).toarray()
    return np.concatenate((x1, x2), axis=1)
