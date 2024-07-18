import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load and Preprocess Data (Example with Placeholder Data)
# Assuming you have a CSV file with columns 'text' (email content) and 'label' (0 for non-spam, 1 for spam)
data = pd.read_csv("your_spam_dataset.csv")

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the ANN Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Hidden Layer
model.add(Dense(1, activation='sigmoid'))  # Output Layer

# 3. Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 5. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Example Usage (Predict on New Email Text)
new_email = ["This is a sample spam email with a fake promotion."]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)[0][0]
if prediction > 0.5:
    print("Spam")
else:
    print("Not Spam")
