import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load dataset
df = pd.read_csv('data/df4_5s.txt')  # make sure delimiter is correct

# 2. Drop columns with all NaN values
df.dropna(axis=1, how='all', inplace=True)

# 3. Drop rows with any NaN values
df.dropna(inplace=True)

# 4. Convert Modifiers to categorical
df['Modifiers'] = df['Modifiers'].astype('category')

# 5. Drop rare classes (e.g., with less than 5 samples)
valid_classes = df['Modifiers'].value_counts()[df['Modifiers'].value_counts() > 5].index
df = df[df['Modifiers'].isin(valid_classes)]

# 6. Define features (exclude ID and Timestamp)
features = df.columns.difference(['Modifiers', 'ID', 'Timestamp'])

# 7. Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 8. Split data
X = df[features]
y = df['Modifiers']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. LDA
lda = LDA()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
print("LDA Accuracy (Test):", accuracy_score(y_test, y_pred_lda))
print(classification_report(y_test, y_pred_lda))

lda_pipe = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
lda_scores = cross_val_score(lda_pipe, X, y, cv=5)
print(f"LDA Accuracy (CV mean): {lda_scores.mean():.4f}")

# 10. QDA
qda_pipe = Pipeline([('scaler', StandardScaler()), ('qda', QDA())])
qda_scores = cross_val_score(qda_pipe, X, y, cv=5)
print(f"QDA Accuracy (CV mean): {qda_scores.mean():.4f}")

# 11. KNN - Cross-validation for different k values
k_range = range(1, 31)
cv_scores = []
f1_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    pipe = Pipeline([('scaler', scaler), ('knn', knn)])
    scores = cross_val_score(pipe, X, y, cv=5)
    cv_scores.append(scores.mean())
    
    # Calculate F1-Score for each k
    y_pred = cross_val_predict(pipe, X, y, cv=5)
    f1 = f1_score(y, y_pred, average='weighted')  # Weighted F1-Score for multi-class classification
    f1_scores.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', label='Accuracy')
plt.plot(k_range, f1_scores, marker='x', label='F1-Score', linestyle='--')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Accuracy and F1-Score for Different k Values')
plt.grid(True)
plt.xticks(k_range)
plt.legend()
plt.show()

# 12. KNN - GridSearch for best k
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
param_grid = {'knn__n_neighbors': list(range(1, 21))}
grid = GridSearchCV(knn_pipe, param_grid, cv=5)
grid.fit(X, y)
print(f"Best KNN Accuracy (CV): {grid.best_score_:.4f}")
print(f" → Best k: {grid.best_params_['knn__n_neighbors']}")

# Final evaluation on test set
best_k = grid.best_params_['knn__n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy (Test):", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Function to plot confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=y_true.cat.categories,
                yticklabels=y_true.cat.categories)
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# Plot confusion matrices
plot_conf_matrix(y_test, y_pred_lda, "LDA")
plot_conf_matrix(y_test, y_pred_knn, f"KNN (k={best_k})")

# Plot comparison of CV scores
models = ['LDA', 'QDA', f'KNN (k={best_k})']
scores = [lda_scores.mean(), qda_scores.mean(), grid.best_score_]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=scores, palette='viridis')
plt.ylim(0, 1)
plt.ylabel("Cross-Validated Accuracy")
plt.title("Model Comparison by Accuracy")
plt.tight_layout()
plt.show()
