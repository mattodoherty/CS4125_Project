from skeleton.skeleton.model.randomforest import RandomForest



def train_and_evaluate_model(X, labels, model_name="RandomForest"):
    # Initialize RandomForest with data
    rf_model = RandomForest(model_name=model_name, embeddings=X, labels=labels)

    # Split data and train
    rf_model.data_transform()
    rf_model.train()

    # Predict and evaluate
    rf_model.predict()
    rf_model.print_results()

    return rf_model.predictions



def train_and_categorize(embeddings, df, category_column):
    from sklearn.model_selection import train_test_split

    # Prepare labels and map them to numeric values
    unique_labels = df[category_column].unique()
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y = df[category_column].map(label_to_index)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=0)

    # Train the model and get predictions
    predictions = train_and_evaluate_model(X_train, y_train)

    # Map predictions back to the original labels
    predicted_labels = [index_to_label[pred] for pred in predictions]

    # Align predictions with the test set rows in the original DataFrame
    categorized_emails = df.loc[
        (df[category_column].map(label_to_index).isin(predictions)) & (df.index.isin(y_test.index))
    ]["Interaction content"].tolist()

    return categorized_emails





