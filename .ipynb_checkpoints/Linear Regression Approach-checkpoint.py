import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv("diabetes.csv")

# Check for duplicated data
data.drop_duplicates(keep="first", inplace=True)

# Feature and target
features = data.drop(["Diabetes"], axis="columns")
target = data["Diabetes"]

# Handling of Categorical Data
cfeatures = pd.get_dummies(features)

# Convert 'YES' and 'NO' to 1 and 0
target = target.replace({'YES': 1, 'NO': 0})

# Train and test split
x_train, x_test, y_train, y_test = train_test_split(cfeatures, target, stratify=target)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Since linear regression is a regression model, you may want to round the predictions to get binary values
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

