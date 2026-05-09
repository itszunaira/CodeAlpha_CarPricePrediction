import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("car data.csv")

# first rows
print("\nFirst 5 Rows:\n")
print(df.head())

# missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# convert categorical data into numbers
df = pd.get_dummies(df, drop_first=True)

# target column
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# graph
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

plt.grid(True)

plt.show()