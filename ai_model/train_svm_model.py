import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load dataset
data = pd.read_csv('data/traffic_logs.csv')

# Define input features and label
X = data[['vehicle_count', 'average_speed', 'waiting_time']]
y = data['congestion']  # Label: 1 = congested, 0 = not congested

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Save the trained model
with open('ai_model/svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ SVM model trained and saved as svm_model.pkl")
