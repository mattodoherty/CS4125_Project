#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
from Config import Config


seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()


    return df


def preprocess_data(filepath: str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(filepath)

    # Ensure data type compatibility
    df['Interaction content'] = df['Interaction content'].values.astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

    # Rename columns for easier handling
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df['Interaction content']
    df["y"] = df["y2"]

    # Remove empty or NaN target labels
    df = df.loc[(df["y"] != '') & (~df["y"].isna())]

    # De-duplicate data (placeholder function)
    df = de_duplication(df)

    # Remove noise from data (placeholder function)
    df = noise_remover(df)

    # Translate data to English
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())

    print(f"Data shape after preprocessing: {df.shape}")
    return df


def get_embeddings(df: pd.DataFrame):
    """
    Generate embeddings using TF-IDF for the relevant columns.
    :param df: Preprocessed DataFrame.
    :return: Combined TF-IDF feature matrix and DataFrame.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_converter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidf_converter.fit_transform(df["Interaction content"]).toarray()
    x2 = tfidf_converter.fit_transform(df["Ticket Summary"]).toarray()
    X = np.concatenate((x1, x2), axis=1)

    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """
    Create and return a Data object for training and testing.
    :param X: Feature matrix.
    :param df: DataFrame containing target labels.
    :return: Data object.
    """
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name: str):
    """
    Perform model training and evaluation.
    :param data: Data object containing training and testing splits.
    :param df: DataFrame containing data.
    :param name: Name of the model.
    """
    model = model_predict(data, df, name)
    model.print_results(data)

# Code will start executing from following line
if __name__ == '__main__':
    # Load and preprocess the data
    df = load_data()

    df = preprocess_data(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Generate embeddings
    X, group_df = get_embeddings(df)

    # Create a Data object
    data = get_data_object(X, df)

    # Perform modelling
    perform_modelling(data, df, 'RandomForest')

