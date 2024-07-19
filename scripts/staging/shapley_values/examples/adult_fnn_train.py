import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

def preprocess_data():
    # Load dataset
    column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                    "hours_per_week", "native_country", "income"]

    df = pd.read_csv('adult.data', names=column_names, sep=r'\s*,\s*', engine='python', na_values="?")
    df.dropna(inplace=True)

    # Encode categorical variables
    categorical_features = ["workclass", "education", "marital_status", "occupation",
                            "relationship", "race", "sex", "native_country"]
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(df[categorical_features])

    # Normalize numerical features
    numerical_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])

    # Combine features
    X = np.hstack([scaled_numerical, encoded_categorical])

    # Encode target variable
    y = (df["income"] == ">50K").astype(np.float32).values

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(input_size, hidden_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='sigmoid'))
    return model

def main():
    # Configuration
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    hidden_size = 128

    # Data preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()

    # Model setup
    input_size = X_train.shape[1]
    output_size = 1  # Binary classification

    model = create_model(input_size, hidden_size, output_size)
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()