import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.utils import resample

# Load and preprocess data
data = pd.read_csv("cre.csv")
x = data.drop(columns=["slno", "last", "first", "lat", "long", "city_pop", "merch_lat", "merch_long", "is_fraud"])

# Replace infinity with NaN
x.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop or fill NaN values
x.dropna(inplace=True)  # or use df.fillna(0) or df.fillna(df.mean())
# Ensure correct data type
x = pd.get_dummies(x)

y = data["is_fraud"]
y = y[x.index]

# Handling class imbalance
x = pd.concat([x, y], axis=1)
fraud = x[x.is_fraud == 1]
not_fraud = x[x.is_fraud == 0]

fraud_upsampled = resample(fraud, replace=True, n_samples=len(not_fraud), random_state=42)
upsampled = pd.concat([not_fraud, fraud_upsampled])

y = upsampled.is_fraud
x = upsampled.drop(columns=["is_fraud"])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest Classifier
r = RandomForestClassifier()
r.fit(x_train, y_train)
y_predict = r.predict(x_test)
ac2 = accuracy_score(y_predict, y_test)
print("Random Forest Accuracy:", ac2)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict, zero_division=1))

# Save the model
joblib.dump(r, "Mymodel.h5")
