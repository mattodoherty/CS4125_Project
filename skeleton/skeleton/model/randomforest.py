import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skeleton.skeleton.model.base import BaseModel

# Random seed for reproducibility
seed = 0
np.random.seed(seed)


class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
        super(RandomForest, self).__init__(model_name=model_name, embeddings=embeddings, y=labels)
        self.model_name = model_name
        self.embeddings = embeddings
        self.labels = labels
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None


    def data_transform(self) -> None:
        print("Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.embeddings, self.labels, test_size=0.2, random_state=seed
        )
        print("Data split completed.")

    def train(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not prepared. Call data_transform() first.")

        print(f"Training {self.model_name}...")
        self.mdl.fit(self.X_train, self.y_train)
        print(f"Training complete for {self.model_name}.")

    def predict(self) -> None:
        if self.X_test is None:
            raise ValueError("Test data is not prepared. Call data_transform() first.")

        print("Predicting...")
        self.predictions = self.mdl.predict(self.X_test)
        print("Prediction complete.")

    def print_results(self) -> None:
        if self.predictions is None:
            raise ValueError("Predictions not available. Call predict() first.")

        print("\nModel Evaluation:")
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.predictions))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions, zero_division=0))

