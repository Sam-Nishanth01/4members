import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import joblib  # for saving the model and encoders

# Load data
df = pd.read_csv('startup_prediction_data.csv')
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical columns
le_category = LabelEncoder()
df['category_code'] = le_category.fit_transform(df['category_code'])

le_status = LabelEncoder()
df['status_encoded'] = le_status.fit_transform(df['status'])  # acquired=0, closed=1

# Features and target
X = df.drop(columns=['name', 'founded_at', 'status', 'status_encoded'])
y = df['status_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE to balance classes
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 250],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 0.9],
    'scale_pos_weight': [1]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Save model and encoders for deployment
joblib.dump(best_model, 'xgb_startup_model.pkl')
joblib.dump(le_category, 'category_encoder.pkl')
joblib.dump(le_status, 'status_encoder.pkl')

# Plot feature importance
plot_importance(best_model, max_num_features=10)
plt.tight_layout()
plt.savefig("feature_importance.png")  # Save plot for web use
plt.close()

# Predict probabilities
y_proba = best_model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
print(f"\nEvaluation at threshold = {threshold}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))
