# %% [markdown]
# # Heart Disease Risk Assessment: A Machine Learning Approach

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# %%
data = pd.read_csv('/N/u/lli4/BigRed200/cardio/cardio_data_processed.csv')

# %%
cardio_data = data.drop('bp_category', axis=1)

# %%
cardio_data = cardio_data.drop('age', axis=1)

# %%
cardio_data = cardio_data.drop('id', axis=1)

# %%
cardio_data.head()

# %%
# Define the mapping for ordinal encoding
bp_category_mapping = {
    'Normal': 1,
    'Elevated': 2,
    'Hypertension Stage 1': 3,
    'Hypertension Stage 2': 4
}

# Perform ordinal encoding on the "bp_category_encoded" column
cardio_data['bp_category_encoded'] = cardio_data['bp_category_encoded'].map(bp_category_mapping)

# Display the first few rows after encoding
cardio_data.head()

# %%
from sklearn.preprocessing import StandardScaler

# Instantiate the StandardScaler
scaler = StandardScaler()

# Apply scaling to the entire dataset excluding the binary columns (since they are already in the range [0, 1])
columns_to_scale = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'age_years', 'bmi']

cardio_data[columns_to_scale] = scaler.fit_transform(cardio_data[columns_to_scale])

# Display the first few rows of the scaled dataset
cardio_data.head()

# %%
features = ["height", "weight", "bmi"]

for feature in features:
    Q1 = cardio_data[feature].quantile(0.25)
    Q3 = cardio_data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a filtered DataFrame without outliers for the current feature
    cardio_data = cardio_data[(cardio_data[feature] >= lower_bound) & (cardio_data[feature] <= upper_bound)]

# %% [markdown]
# ## Baseline Model

# %% [markdown]
# - Splitting the data set in to train, validation, and test set.
#   - First, the dataset is split into a 60% training set and a 40% temporary set.
#   - Then, the temporary set is equally divided into validation and test sets, each consisting of 20% of the original data.

# %%
# Entire dataset
X_all = cardio_data.drop('cardio', axis=1)
y_all = cardio_data['cardio']
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# %%
X_train_all.shape, X_val_all.shape, X_test_all.shape, y_train_all.shape, y_val_all.shape, y_test_all.shape

# %%
# Women's data
women_data = cardio_data[cardio_data['gender'] == 1]
X_women = women_data.drop('cardio', axis=1)
X_women = X_women.drop('gender', axis=1)
y_women = women_data['cardio']
X_train_women, X_test_women, y_train_women, y_test_women = train_test_split(X_women, y_women, test_size=0.2, random_state=42)
X_train_women, X_val_women, y_train_women, y_val_women = train_test_split(X_train_women, y_train_women, test_size=0.2, random_state=42)

# %%
X_train_women.shape, X_val_women.shape, X_test_women.shape, y_train_women.shape, y_val_women.shape, y_test_women.shape

# %%
# Men's data
men_data = cardio_data[cardio_data['gender'] == 2]
X_men = men_data.drop('cardio', axis=1)
X_men = X_men.drop('gender', axis=1)
y_men = men_data['cardio']
# Split men's data into train, validation, and test sets
X_train_men, X_test_men, y_train_men, y_test_men = train_test_split(X_men, y_men, test_size=0.2, random_state=42)
X_train_men, X_val_men, y_train_men, y_val_men = train_test_split(X_train_men, y_train_men, test_size=0.2, random_state=42)

# # %%
# X_train_men.shape, X_val_men.shape, X_test_men.shape, y_train_men.shape, y_val_men.shape, y_test_men.shape

# # %%
# # Initialize the Logistic Regression model
# lr_all = LogisticRegression(max_iter=1000)
# lr_women = LogisticRegression(max_iter=1000)
# lr_men = LogisticRegression(max_iter=1000)

# # Training and evaluating for the entire dataset
# lr_all.fit(X_train_all, y_train_all)
# y_pred_val_all = lr_all.predict(X_val_all)

# # Training and evaluating for women's data
# lr_women.fit(X_train_women, y_train_women)
# y_pred_val_women = lr_women.predict(X_val_women)

# # Training and evaluating for men's data
# lr_men.fit(X_train_men, y_train_men)
# y_pred_val_men = lr_men.predict(X_val_men)

# # %%
# with open('lr_model_all.pkl', 'wb') as file:
#     pickle.dump(lr_all, file)

# with open('lr_model_women.pkl', 'wb') as file:
#     pickle.dump(lr_women, file)

# with open('lr_model_men.pkl', 'wb') as file:
#     pickle.dump(lr_men, file)

# print("Model saved successfully!")

# %%
# Define a function to compute and display the metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    print(f"------ {model_name} ------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion)
    print("\n")

# # Evaluating the models
# evaluate_model(y_val_all, y_pred_val_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_val_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_val_men, "Men's Data")

# # %% [markdown]
# # # Applying Machin Learning Models
# # ## 1. Support Vector Machine (SVM)

# # %%
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV

# # %%
# # Define the SVM and KNN models
# svm_model = SVC()

# # Training and evaluating SVM on different datasets
# # Entire dataset
# svm_model.fit(X_train_all, y_train_all)
# svm_pred_all = svm_model.predict(X_val_all)

# # Women's data
# svm_model.fit(X_train_women, y_train_women)
# svm_pred_Women = svm_model.predict(X_val_women)

# # Men's data
# svm_model.fit(X_train_men, y_train_men)
# svm_pred_men = svm_model.predict(X_val_men)

# # Evaluating the models
# print("Support Vector Machine (SVM):")
# evaluate_model(y_val_all, svm_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, svm_pred_Women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, svm_pred_men, "Men's Data")

# # %% [markdown]
# # ## 2. K-Nearest Neighbors (K-NN)

# # %%
# knn_model = KNeighborsClassifier()

# # Training and evaluating KNN on different datasets
# # Entire dataset
# knn_model.fit(X_train_all, y_train_all)
# knn_pred_all = knn_model.predict(X_val_all)

# # Women's data
# knn_model.fit(X_train_women, y_train_women)
# knn_pred_Women = knn_model.predict(X_val_women)

# # Men's data
# knn_model.fit(X_train_men, y_train_men)
# knn_pred_men = knn_model.predict(X_val_men)

# # Evaluating the models
# print("K-Nearest Neighbors (K-NN):")
# evaluate_model(y_val_all, knn_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, knn_pred_Women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, knn_pred_men, "Men's Data")

# # %% [markdown]
# # # 3. XGBoost

# # %%
# import xgboost as xgb

# # %%
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# # Entire dataset
# xgb_model.fit(X_train_all, y_train_all)
# xgb_pred_all = xgb_model.predict(X_val_all)

# # Women's data
# xgb_model.fit(X_train_women, y_train_women)
# xgb_pred_Women = xgb_model.predict(X_val_women)

# # Men's data
# xgb_model.fit(X_train_men, y_train_men)
# xgb_pred_men = xgb_model.predict(X_val_men)

# # Evaluating the models
# print("XGBoost Model:")
# evaluate_model(y_val_all, xgb_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, xgb_pred_Women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, xgb_pred_men, "Men's Data")

# # %% [markdown]
# # ## 4. Decision Tree

# # %%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Initialize the Decision Tree model
# dt = DecisionTreeClassifier(random_state=42)

# # Train the model on the entire dataset
# dt.fit(X_train_all, y_train_all)
# dt_pred_all = dt.predict(X_val_all)

# # Train the model on the women's dataset
# dt.fit(X_train_women, y_train_women)
# dt_pred_women = dt.predict(X_val_women)

# # Train the model on the men's dataset
# dt.fit(X_train_men, y_train_men)
# dt_pred_men = dt.predict(X_val_men)

# # Evaluating the models
# print("Decision Tree:")
# evaluate_model(y_val_all, dt_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, dt_pred_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, dt_pred_men, "Men's Data")

# # %% [markdown]
# # ## 5. Random Forest

# # %%
# from sklearn.ensemble import RandomForestClassifier

# # Initialize the Random Forest model
# rf = RandomForestClassifier(random_state=42)

# # Train the model on the entire dataset
# rf.fit(X_train_all, y_train_all)
# rf_pred_all = rf.predict(X_val_all)

# # Train the model on the women's dataset
# rf.fit(X_train_women, y_train_women)
# rf_pred_women = rf.predict(X_val_women)

# # Train the model on the men's dataset
# rf.fit(X_train_men, y_train_men)
# rf_pred_men = rf.predict(X_val_men)

# # Evaluating the models
# print("Random Forest:")
# evaluate_model(y_val_all, rf_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, rf_pred_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, rf_pred_men, "Men's Data")

# # %% [markdown]
# # ## 8. Naive Bayesian

# # %%
# from sklearn.naive_bayes import GaussianNB

# # Initialize the Naive Bayes model
# nb = GaussianNB()

# # Train the model on the entire dataset
# nb.fit(X_train_all, y_train_all)
# nb_pred_all = nb.predict(X_val_all)

# # Train the model on the women's dataset
# nb.fit(X_train_women, y_train_women)
# nb_pred_women = nb.predict(X_val_women)

# # Train the model on the men's dataset
# nb.fit(X_train_men, y_train_men)
# nb_pred_men = nb.predict(X_val_men)

# # Evaluating the models
# print("Naive Bayes:")
# evaluate_model(y_val_all, nb_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, nb_pred_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, nb_pred_men, "Men's Data")

# # %% [markdown]
# # ## 7. Perceptron Neuron

# # %%
# from sklearn.linear_model import Perceptron

# # Initialize the Perceptron model
# perceptron = Perceptron(random_state=42)

# # Train the model on the entire dataset
# perceptron.fit(X_train_all, y_train_all)
# perceptron_pred_all = perceptron.predict(X_val_all)

# # Train the model on the women's dataset
# perceptron.fit(X_train_women, y_train_women)
# perceptron_pred_women = perceptron.predict(X_val_women)

# # Train the model on the men's dataset
# perceptron.fit(X_train_men, y_train_men)
# perceptron_pred_men = perceptron.predict(X_val_men)

# # Evaluating the models
# print("Perceptron:")
# evaluate_model(y_val_all, perceptron_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, perceptron_pred_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, perceptron_pred_men, "Men's Data")

# # %% [markdown]
# # ## 8. Multi-Layer Perceptron (MLP)

# # %%
# from sklearn.neural_network import MLPClassifier

# # Initialize the MLP model
# mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)

# # Train the model on the entire dataset
# mlp.fit(X_train_all, y_train_all)
# mlp_pred_all = mlp.predict(X_val_all)

# # Train the model on the women's dataset
# mlp.fit(X_train_women, y_train_women)
# mlp_pred_women = mlp.predict(X_val_women)

# # Train the model on the men's dataset
# mlp.fit(X_train_men, y_train_men)
# mlp_pred_men = mlp.predict(X_val_men)

# # Evaluating the models
# print("Multi-Layer Perceptron (MLP):")
# evaluate_model(y_val_all, mlp_pred_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, mlp_pred_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, mlp_pred_men, "Men's Data")

# # %% [markdown]
# # #Hyperparameter Tuning to Both models
# # 
# # ---
# # 
# # 

# # %% [markdown]
# # ## 1. SVM with Hyperparameter Tuning

# # %%
# # Hyperparameter grids
# svm_param_grid = {
#     'C': [1, 10],
#     'gamma': ['scale'],
#     'kernel': ['rbf']
# }

# # Grid search setup
# svm_grid_search_all = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1)
# svm_grid_search_women = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1)
# svm_grid_search_men = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1)

# # %%
# svm_grid_search_all.fit(X_train_all, y_train_all)
# best_params_svm_all = SVC(**svm_grid_search_all.best_params_)
# best_svm_model_all = svm_grid_search_all.best_estimator_

# # %%
# import pickle

# svm_model_all = best_svm_model_all

# with open('svm_model_all.pkl', 'wb') as file:
#     pickle.dump(svm_model_all, file)

# print("Model saved successfully!")

# # %%
# svm_grid_search_women.fit(X_train_women, y_train_women)
# best_params_svm_women = SVC(**svm_grid_search_women.best_params_)
# best_svm_model_women = svm_grid_search_women.best_estimator_

# # %%
# svm_model_women = best_svm_model_women

# with open('svm_model_women.pkl', 'wb') as file:
#     pickle.dump(svm_model_women, file)

# print("Model saved successfully!")

# # %%
# svm_grid_search_men.fit(X_train_men, y_train_men)
# best_params_svm_men = SVC(**svm_grid_search_men.best_params_)
# best_svm_model_men = svm_grid_search_men.best_estimator_

# # %%
# svm_model_men = best_svm_model_men

# with open('svm_model_men.pkl', 'wb') as file:
#     pickle.dump(svm_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_svm_tuned_all = best_svm_model_all.predict(X_val_all)

# # Women's Data
# y_pred_svm_tuned_women = best_svm_model_women.predict(X_val_women)

# # Men's Data
# y_pred_svm_tuned_men = best_svm_model_men.predict(X_val_men)

# # Evaluating the models
# print("SVM Model:")
# evaluate_model(y_val_all, y_pred_svm_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_svm_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_svm_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 2. KNN with Hyperparameter Tuning

# # %%
# knn_param_grid = {
#     'n_neighbors': [3, 5, 7],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

# knn_grid_search_all = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1)
# knn_grid_search_women = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1)
# knn_grid_search_men = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1)

# # %%
# knn_grid_search_all.fit(X_train_all, y_train_all)
# best_params_svm_all = KNeighborsClassifier(**knn_grid_search_all.best_params_)
# best_knn_model_all = knn_grid_search_all.best_estimator_

# # %%
# knn_model_all = best_knn_model_all

# with open('knn_model_all.pkl', 'wb') as file:
#     pickle.dump(knn_model_all, file)

# print("Model saved successfully!")

# # %%
# knn_grid_search_women.fit(X_train_women, y_train_women)
# best_params_knn_women = KNeighborsClassifier(**knn_grid_search_women.best_params_)
# best_knn_model_women = knn_grid_search_women.best_estimator_

# # %%
# knn_model_women = best_knn_model_women

# with open('knn_model_women.pkl', 'wb') as file:
#     pickle.dump(knn_model_women, file)

# print("Model saved successfully!")

# # %%
# knn_grid_search_men.fit(X_train_men, y_train_men)
# best_params_knn_men = KNeighborsClassifier(**knn_grid_search_men.best_params_)
# best_knn_model_men = knn_grid_search_men.best_estimator_

# # %%
# knn_model_men = best_knn_model_men

# with open('knn_model_men.pkl', 'wb') as file:
#     pickle.dump(knn_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_knn_tuned_all = best_knn_model_all.predict(X_val_all)

# # Women's Data
# y_pred_knn_tuned_women = best_knn_model_women.predict(X_val_women)

# # Men's Data
# y_pred_knn_tuned_men = best_knn_model_men.predict(X_val_men)

# # Evaluating the models
# print("KNN Model:")
# evaluate_model(y_val_all, y_pred_knn_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_knn_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_knn_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 3. XGBoost with Hyperparameter Tuning

# # %%
# xgb_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1],
#     'colsample_bytree': [0.8, 1]
# }

# xgb_grid_search_all = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=5, n_jobs=-1)
# xgb_grid_search_women = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=5, n_jobs=-1)
# xgb_grid_search_men = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=5, n_jobs=-1)

# # %%
# xgb_grid_search_all.fit(X_train_all, y_train_all)
# best_xgb_params_all = xgb_grid_search_all.best_params_
# best_xgb_model_all = xgb_grid_search_all.best_estimator_

# # %%
# xgb_model_all = best_xgb_model_all

# with open('xgb_model_all.pkl', 'wb') as file:
#     pickle.dump(xgb_model_all, file)

# print("Model saved successfully!")

# # %%
# xgb_grid_search_women.fit(X_train_women, y_train_women)
# best_xgb_params_women = xgb_grid_search_women.best_params_
# best_xgb_model_women = xgb_grid_search_women.best_estimator_

# # %%
# xgb_model_women = best_xgb_model_women

# with open('xgb_model_women.pkl', 'wb') as file:
#     pickle.dump(xgb_model_women, file)

# print("Model saved successfully!")

# # %%
# xgb_grid_search_men.fit(X_train_men, y_train_men)
# best_xgb_params_men = xgb_grid_search_men.best_params_
# best_xgb_model_men = xgb_grid_search_men.best_estimator_

# # %%
# xgb_model_men = best_xgb_model_men

# with open('xgb_model_men.pkl', 'wb') as file:
#     pickle.dump(xgb_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_xgb_tuned_all = best_xgb_model_all.predict(X_val_all)

# # Women's Data
# y_pred_xgb_tuned_women = best_xgb_model_women.predict(X_val_women)

# # Men's Data
# y_pred_xgb_tuned_men = best_xgb_model_men.predict(X_val_men)

# # Evaluating the models
# print("XGBoost Model:")
# evaluate_model(y_val_all, y_pred_xgb_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_xgb_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_xgb_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 4. Decision Tree with Hyperparameter Tuning

# # %%
# # Decision Tree Parameter Grid
# param_grid = {
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'criterion': ['gini', 'entropy']
# }

# dt_grid_search_all = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# dt_grid_search_women = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# dt_grid_search_men = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # %%
# dt_grid_search_all.fit(X_train_all, y_train_all)
# best_dt_params_all = dt_grid_search_all.best_params_
# best_dt_model_all = dt_grid_search_all.best_estimator_

# # %%
# dt_model_all = best_dt_model_all

# with open('dt_model_all.pkl', 'wb') as file:
#     pickle.dump(dt_model_all, file)

# print("Model saved successfully!")

# # %%
# dt_grid_search_women.fit(X_train_women, y_train_women)
# best_dt_params_women = dt_grid_search_women.best_params_
# best_dt_model_women = dt_grid_search_women.best_estimator_

# # %%
# dt_model_women = best_dt_model_women

# with open('dt_model_women.pkl', 'wb') as file:
#     pickle.dump(dt_model_women, file)

# print("Model saved successfully!")

# # %%
# dt_grid_search_men.fit(X_train_men, y_train_men)
# best_dt_params_men = dt_grid_search_men.best_params_
# best_dt_model_men = dt_grid_search_men.best_estimator_

# # %%
# dt_model_men = best_dt_model_men

# with open('dt_model_men.pkl', 'wb') as file:
#     pickle.dump(dt_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_dt_tuned_all = best_dt_model_all.predict(X_val_all)

# # Women's Data
# y_pred_dt_tuned_women = best_dt_model_women.predict(X_val_women)

# # Men's Data
# y_pred_dt_tuned_men = best_dt_model_men.predict(X_val_men)

# # Evaluating the models
# print("Decision Tree:")
# evaluate_model(y_val_all, y_pred_dt_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_dt_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_dt_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 5. Random Forest with Hyperparameter Tuning
# # 
# # 
# # 

# # %%
# # Random Forest Parameter Grid
# rf_param_grid = {
#     'n_estimators': [50, 100],
#     'max_depth': [10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['auto', 'sqrt'],
#     'bootstrap': [True, False]
# }

# rf_grid_search_all = GridSearchCV(rf, rf_param_grid, cv=10, scoring='accuracy', n_jobs=-1)
# rf_grid_search_women = GridSearchCV(rf, rf_param_grid, cv=10, scoring='accuracy', n_jobs=-1)
# rf_grid_search_men = GridSearchCV(rf, rf_param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# # %%
# rf_grid_search_all.fit(X_train_all, y_train_all)
# best_rf_params_all = rf_grid_search_all.best_params_
# best_rf_model_all = rf_grid_search_all.best_estimator_

# # %%
# rf_model_all = best_rf_model_all

# with open('rf_model_all.pkl', 'wb') as file:
#     pickle.dump(rf_model_all, file)

# print("Model saved successfully!")

# # %%
# rf_grid_search_women.fit(X_train_women, y_train_women)
# best_rf_params_women = rf_grid_search_women.best_params_
# best_rf_model_women = rf_grid_search_women.best_estimator_

# # %%
# rf_model_women = best_rf_model_women

# with open('rf_model_women.pkl', 'wb') as file:
#     pickle.dump(rf_model_women, file)

# print("Model saved successfully!")

# # %%
# rf_grid_search_men.fit(X_train_men, y_train_men)
# best_rf_params_men = rf_grid_search_men.best_params_
# best_rf_model_men = rf_grid_search_men.best_estimator_

# # %%
# rf_model_men = best_rf_model_men

# with open('rf_model_men.pkl', 'wb') as file:
#     pickle.dump(rf_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_rf_tuned_all = best_rf_model_all.predict(X_val_all)

# # Women's Data
# y_pred_rf_tuned_women = best_rf_model_women.predict(X_val_women)

# # Men's Data
# y_pred_rf_tuned_men = best_rf_model_men.predict(X_val_men)

# # Evaluating the models
# print("Random Forest:")
# evaluate_model(y_val_all, y_pred_rf_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_rf_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_rf_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 6. Naive Bayes with Hyperparameter Tuning

# # %%
# # Naive Bayes Parameter Grid
# gnb_param_grid = {
#     'var_smoothing': [1e-9, 1e-8, 1e-7]  # Example values for var_smoothing
# }

# gnb_grid_search_all = GridSearchCV(GaussianNB(), gnb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# gnb_grid_search_women = GridSearchCV(GaussianNB(), gnb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# gnb_grid_search_men = GridSearchCV(GaussianNB(), gnb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # %%
# gnb_grid_search_all.fit(X_train_all, y_train_all)
# best_gnb_params_all = gnb_grid_search_all.best_params_
# best_gnb_model_all = gnb_grid_search_all.best_estimator_

# # %%
# gnb_model_all = best_gnb_model_all

# with open('gnb_model_all.pkl', 'wb') as file:
#     pickle.dump(gnb_model_all, file)

# print("Model saved successfully!")

# # %%
# gnb_grid_search_women.fit(X_train_women, y_train_women)
# best_gnb_params_women = gnb_grid_search_women.best_params_
# best_gnb_model_women = gnb_grid_search_women.best_estimator_

# # %%
# gnb_model_women = best_gnb_model_women

# with open('gnb_model_women.pkl', 'wb') as file:
#     pickle.dump(gnb_model_women, file)

# print("Model saved successfully!")

# # %%
# gnb_grid_search_men.fit(X_train_men, y_train_men)
# best_gnb_params_men = gnb_grid_search_men.best_params_
# best_gnb_model_men = gnb_grid_search_men.best_estimator_

# # %%
# gnb_model_men = best_gnb_model_men

# with open('gnb_model_men.pkl', 'wb') as file:
#     pickle.dump(gnb_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_gnb_tuned_all = best_gnb_model_all.predict(X_val_all)

# # Women's Data
# y_pred_gnb_tuned_women = best_gnb_model_women.predict(X_val_women)

# # Men's Data
# y_pred_gnb_tuned_men = best_gnb_model_men.predict(X_val_men)

# # Evaluating the models
# print("Naive Bayes:")
# evaluate_model(y_val_all, y_pred_gnb_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_gnb_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_gnb_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 7. Perceptron with Hyperparameter Tuning

# # %%
# # Perceptron Parameter Grid
# perceptron_param_grid = {
#     'eta0': [0.1, 1, 10]  # Example values for learning rate (eta0)
# }

# perceptron_grid_search_all = GridSearchCV(Perceptron(max_iter=10000), perceptron_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# perceptron_grid_search_women = GridSearchCV(Perceptron(max_iter=10000), perceptron_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# perceptron_grid_search_men = GridSearchCV(Perceptron(max_iter=10000), perceptron_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # %%
# perceptron_grid_search_all.fit(X_train_all, y_train_all)
# best_perceptron_params_all = perceptron_grid_search_all.best_params_
# best_perceptron_model_all = perceptron_grid_search_all.best_estimator_

# # %%
# perceptron_model_all = best_perceptron_model_all

# with open('perceptron_model_all.pkl', 'wb') as file:
#     pickle.dump(perceptron_model_all, file)

# print("Model saved successfully!")

# # %%
# perceptron_grid_search_women.fit(X_train_women, y_train_women)
# best_perceptron_params_women = perceptron_grid_search_women.best_params_
# best_perceptron_model_women = perceptron_grid_search_women.best_estimator_

# # %%
# perceptron_model_women = best_perceptron_model_women

# with open('perceptron_model_women.pkl', 'wb') as file:
#     pickle.dump(perceptron_model_women, file)

# print("Model saved successfully!")

# # %%
# perceptron_grid_search_men.fit(X_train_men, y_train_men)
# best_perceptron_params_men = perceptron_grid_search_men.best_params_
# best_perceptron_model_men = perceptron_grid_search_men.best_estimator_

# # %%
# perceptron_model_men = best_perceptron_model_men

# with open('perceptron_model_men.pkl', 'wb') as file:
#     pickle.dump(perceptron_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_perceptron_tuned_all = best_perceptron_model_all.predict(X_val_all)

# # Women's Data
# y_pred_perceptron_tuned_women = best_perceptron_model_women.predict(X_val_women)

# # Men's Data
# y_pred_perceptron_tuned_men = best_perceptron_model_men.predict(X_val_men)

# # Evaluating the models
# print("Perceptron:")
# evaluate_model(y_val_all, y_pred_perceptron_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_perceptron_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_perceptron_tuned_men, "Men's Data")

# # %% [markdown]
# # ## 8. Multi-Layer Perceptron (MLP)

# # %%
# # MLP Parameter Grid
# mlp_param_grid = {
#     'hidden_layer_sizes': [(50, 50)],
#     'learning_rate_init': [0.01]
# }

# mlp_grid_search_all = GridSearchCV(MLPClassifier(max_iter=1000, random_state=1), mlp_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# mlp_grid_search_women = GridSearchCV(MLPClassifier(max_iter=1000, random_state=1), mlp_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# mlp_grid_search_men = GridSearchCV(MLPClassifier(max_iter=1000, random_state=1), mlp_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # %%
# mlp_grid_search_all.fit(X_train_all, y_train_all)
# best_mlp_params_all = mlp_grid_search_all.best_params_
# best_mlp_model_all = mlp_grid_search_all.best_estimator_

# # %%
# mlp_model_all = best_mlp_model_all

# with open('mlp_model_all.pkl', 'wb') as file:
#     pickle.dump(mlp_model_all, file)

# print("Model saved successfully!")

# # %%
# mlp_grid_search_women.fit(X_train_women, y_train_women)
# best_mlp_params_women = mlp_grid_search_women.best_params_
# best_mlp_model_women = mlp_grid_search_women.best_estimator_

# # %%
# mlp_model_women = best_mlp_model_women

# with open('mlp_model_women.pkl', 'wb') as file:
#     pickle.dump(mlp_model_women, file)

# print("Model saved successfully!")

# # %%
# mlp_grid_search_men.fit(X_train_men, y_train_men)
# best_mlp_params_men = mlp_grid_search_men.best_params_
# best_mlp_model_men = mlp_grid_search_men.best_estimator_

# # %%
# mlp_model_men = best_mlp_model_men

# with open('mlp_model_men.pkl', 'wb') as file:
#     pickle.dump(mlp_model_men, file)

# print("Model saved successfully!")

# # %%
# # Entire Data
# y_pred_mlp_tuned_all = best_mlp_model_all.predict(X_val_all)

# # Women's Data
# y_pred_mlp_tuned_women = best_mlp_model_women.predict(X_val_women)

# # Men's Data
# y_pred_mlp_tuned_men = best_mlp_model_men.predict(X_val_men)

# # Evaluating the models
# print("Multi-Layer Perceptron (MLP):")
# evaluate_model(y_val_all, y_pred_mlp_tuned_all, "Entire Data")
# if not X_train_women.empty:
#     evaluate_model(y_val_women, y_pred_mlp_tuned_women, "Women's Data")
# if not X_train_men.empty:
#     evaluate_model(y_val_men, y_pred_mlp_tuned_men, "Men's Data")

# %% [markdown]
# # Ensemble Model

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier

# %%
# Load the trained models
with open('/N/u/lli4/BigRed200/cardio/rf_model_all.pkl', 'rb') as file:
    model_all_1 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/svm_model_all.pkl', 'rb') as file:
    model_all_2 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/xgb_model_all.pkl', 'rb') as file:
    model_all_3 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/lr_model_all.pkl', 'rb') as file:
    model_all_4 = pickle.load(file)

with open('/N/u/lli4/BigRed200/cardio/rf_model_women.pkl', 'rb') as file:
    model_women_1 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/svm_model_women.pkl', 'rb') as file:
    model_women_2 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/xgb_model_women.pkl', 'rb') as file:
    model_women_3 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/lr_model_women.pkl', 'rb') as file:
    model_women_4 = pickle.load(file)

with open('/N/u/lli4/BigRed200/cardio/rf_model_men.pkl', 'rb') as file:
    model_men_1 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/svm_model_men.pkl', 'rb') as file:
    model_men_2 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/xgb_model_men.pkl', 'rb') as file:
    model_men_3 = pickle.load(file)
with open('/N/u/lli4/BigRed200/cardio/lr_model_men.pkl', 'rb') as file:
    model_men_4 = pickle.load(file)

# # %%
# # Ensemble model for the entire dataset with weighted average
# ensemble_all = VotingClassifier(
#     estimators=[
#         ('rf', model_all_1),
#         ('svm', model_all_2),
#         ('xgb', model_all_3),
#         ('lr', model_all_4)
#     ],
#     voting='hard',
#     weights=[2, 1, 3, 1]
# )

# # Ensemble model for women's dataset with weighted average
# ensemble_women = VotingClassifier(
#     estimators=[
#         ('rf', model_women_1),
#         ('svm', model_women_2),
#         ('xgb', model_women_3),
#         ('lr', model_women_4)
#     ],
#     voting='hard',
#     weights=[2, 1, 3, 1]
# )

# # Ensemble model for men's dataset with weighted average
# ensemble_men = VotingClassifier(
#     estimators=[
#         ('rf', model_men_1),
#         ('svm', model_men_2),
#         ('xgb', model_men_3),
#         ('lr', model_men_4)
#     ],
#     voting='hard',
#     weights=[2, 1, 3, 1]
# )


# # %%
# # Fit and evaluate ensemble model for the entire dataset
# ensemble_all.fit(X_train_all, y_train_all)
# predictions_all = ensemble_all.predict(X_test_all)
# evaluate_model(y_test_all, predictions_all, "Ensemble - Entire Data")

# # Fit and evaluate ensemble model for women's dataset
# ensemble_women.fit(X_train_women, y_train_women)
# predictions_women = ensemble_women.predict(X_test_women)
# evaluate_model(y_test_women, predictions_women, "Ensemble - Women's Data")

# # Fit and evaluate ensemble model for men's dataset
# ensemble_men.fit(X_train_men, y_train_men)
# predictions_men = ensemble_men.predict(X_test_men)
# evaluate_model(y_test_men, predictions_men, "Ensemble - Men's Data")

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize individual models for the entire dataset (weak learners)
rf_weak_all = model_all_1
svm_weak_all = model_all_2 
xgb_weak_all = model_all_3
lr_weak_all = model_all_4
 # Note: For AdaBoost, the base estimator needs to support probability estimates

# Initialize individual models for the women's dataset (weak learners)
rf_weak_women = model_women_1
svm_weak_women = model_women_2 
xgb_weak_women = model_women_3
lr_weak_women = model_women_4

# Initialize individual models for the men's dataset (weak learners)
rf_weak_men = model_men_1
svm_weak_men = model_men_2 
xgb_weak_men = model_men_3
lr_weak_men = model_men_4

# Define the AdaBoost classifier for the entire dataset
ada_boost_all = AdaBoostClassifier(
    base_estimator=None,
    algorithm='SAMME',
    random_state=42
)

# Define the AdaBoost classifier for the women's dataset
ada_boost_women = AdaBoostClassifier(
    base_estimator=None,
    random_state=42,
    algorithm='SAMME'
)

# Define the AdaBoost classifier for the men's dataset
ada_boost_men = AdaBoostClassifier(
    base_estimator=None,
    random_state=42,
    algorithm='SAMME'
)

# Define hyperparameters to tune for AdaBoost for the entire dataset
param_grid_all = {
    'n_estimators': [10, 20],  # Number of weak learners
    'learning_rate': [0.01,0.1]  # Learning rate
}

# Define hyperparameters to tune for AdaBoost for the women's dataset
param_grid_women = {
    'n_estimators': [10, 20],  # Number of weak learners
    'learning_rate': [0.01,0.1]  # Learning rate
}

# Define hyperparameters to tune for AdaBoost for the men's dataset
param_grid_men = {
    'n_estimators': [10, 20],  # Number of weak learners
    'learning_rate': [0.01,0.1]  # Learning rate
}

datasets = {
    "Entire Dataset": (X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all),
    "Women's Dataset": (X_train_women, y_train_women, X_val_women, y_val_women, X_test_women, y_test_women),
    "Men's Dataset": (X_train_men, y_train_men, X_val_men, y_val_men, X_test_men, y_test_men)
}

for dataset_name, (X_train, y_train, X_val, y_val, X_test, y_test) in datasets.items():
    if dataset_name == "Entire Dataset":
        weak_learners = [
            ("Logistic Regression", lr_weak_all),
            ("SVM", svm_weak_all),
            ("Random Forest", rf_weak_all),
            ("XGBoost", xgb_weak_all)
        ]
        ada_boost = ada_boost_all
        param_grid = param_grid_all
    elif dataset_name == "Women's Dataset":
        weak_learners = [
            ("Logistic Regression", lr_weak_women),
            ("SVM", svm_weak_women),
            ("Random Forest", rf_weak_women),
            ("XGBoost", xgb_weak_women)
        ]
        ada_boost = ada_boost_women
        param_grid = param_grid_women
    elif dataset_name == "Men's Dataset":
        weak_learners = [
            ("Logistic Regression", lr_weak_men),
            ("SVM", svm_weak_men),
            ("Random Forest", rf_weak_men),
            ("XGBoost", xgb_weak_men)
        ]
        ada_boost = ada_boost_men
        param_grid = param_grid_men
    
    for learner_name, weak_learner in weak_learners:
        ada_boost.base_estimator = weak_learner

        # Create GridSearchCV for AdaBoost hyperparameters
        grid_search = GridSearchCV(ada_boost, param_grid, cv=5)
        grid_search.fit(X_val, y_val)
        best_ada_boost = grid_search.best_estimator_

        # Fit the best AdaBoost model
        best_ada_boost.fit(X_train, y_train)

        # Make predictions and evaluate on the test set
        predictions_test = best_ada_boost.predict(X_test)
        evaluate_model(y_test, predictions_test, f"{dataset_name} (AdaBoost with {learner_name})")


