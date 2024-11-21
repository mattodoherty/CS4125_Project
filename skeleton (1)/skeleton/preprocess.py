#Methods related to data loading and all pre-processing steps will go here

import pandas as pd
import re
from googletrans import Translator
from Config import Config


def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame based on important columns.
    :param df: Input DataFrame.
    :return: Deduplicated DataFrame.
    """
    return df.drop_duplicates(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT], keep='first')


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean noise (like special characters, excessive whitespace) from text data.
    :param df: Input DataFrame.
    :return: Cleaned DataFrame.
    """

    def clean_text(text: str) -> str:
        # Replace multiple spaces with a single space and remove special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9.,?!\s]', '', text)
        print("Cleaning up...")
        return text.strip()

    # Apply cleaning to the required columns
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)
    return df


def translate_to_en(text_list: list) -> list:
    """
    Translate a list of text entries into English.
    :param text_list: List of strings to be translated.
    :return: List of translated strings.
    """
    translator = Translator()
    translated = []
    for text in text_list:
        try:
            # Translate each text entry and append the result
            translated.append(translator.translate(text, src='auto', dest='en').text)
        except Exception as e:
            # Log the error and keep the original text if translation fails
            print(f"Translation error for text: {text}. Error: {e}")
            translated.append(text)
    return translated


def get_input_data(file_path: str = "data_uef/AppGallery_done.csv") -> pd.DataFrame:
    """
    Load input data from a CSV file.
    :param file_path: Path to the input CSV file.
    :return: Loaded DataFrame.
    """
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
