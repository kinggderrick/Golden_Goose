import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class MachineLearningModel:
    def __init__(self, data, target_column, test_size=0.2, random_state=42):
        """
        Initializes the MachineLearningModel with market data and parameters.

        :param data: DataFrame containing historical market data with features and target.
        :param target_column: Name of the column to predict (e.g., 'PriceUp').
        :param test_size: Proportion of data to use for testing.
        :param random_state: Seed for random number generator.
        """
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)
        self.scaler = StandardScaler()
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares data by handling missing values, scaling features, and splitting into train/test sets.
        """
        # Handle missing values
        self.data.dropna(inplace=True)

        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self):
        """
        Trains the machine learning model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the model's performance on the test data.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def predict(self, X):
        """
        Predicts the target variable for new data.

        :param X: DataFrame or array-like, shape (n_samples, n_features)
        :return: Predicted values
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
