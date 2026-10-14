import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load Dataset
# Replace 'student_health.csv' with your dataset file path
df = pd.read_csv('student_health.csv')

# 2. Define Features (X) and Target Variable (y)
X = df[
    [
        'age',
        'study_hours',
        'sleep_hours',
        'academic_pressure',
        'physical_activity',
    ]
]
y = df['stress_level']

# 3. Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# 6. Save Model to File
joblib.dump(model, 'stress_predictor_model.pkl')
print('Model saved successfully as stress_predictor_model.pkl')