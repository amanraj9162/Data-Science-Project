# Here's a Python snippet that demonstrates the use of machine learning, data visualization, and data manipulation,
#  which are key skills for a data scientist. This code builds a simple classification model using the Random Forest algorithm and visualizes the feature importances, 
# making it impactful for understanding which features are most influential in the predictions.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data.csv')

# Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance Visualization
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance in Random Forest Model')
plt.show()



# Explanation:
# 1. Data Loading & Preprocessing: The code loads a dataset and prepares the features (X) and target variable (y) for training.
# 2. Train-Test Split: It splits the data into training and testing sets to evaluate the model's performance.
# 3. Model Training: The Random Forest Classifier is trained on the training data.
# 4. Evaluation: The code provides an accuracy score and a detailed classification report to assess the model's performance.
# 5. Visualization: The feature importance plot highlights the most influential features, providing insights that are valuable
# for model interpretation.


# This snippet showcases key data science skills: data manipulation, machine learning, model evaluation, and visualization.