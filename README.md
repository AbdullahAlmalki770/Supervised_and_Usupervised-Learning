# ==============================
# Machine Learning Demo:
# Supervised & Unsupervised Learning
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from tensorflow.keras.datasets import mnist


# ---------------------------------
# 1) Supervised Learning – Iris Dataset
# ---------------------------------

print("\n==============================")
print(" 1) Supervised Learning – Iris")
print("==============================\n")

iris = load_iris()
X_iris = iris.data        # المميزات (features)
y_iris = iris.target      # الفئة (label)

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# ---- KNN ----
print("=== Iris – KNN Classifier ===")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print("R2:", r2_score(y_test, y_pred_knn))
print()

# ---- SVM ----
print("=== Iris – SVM Classifier ===")
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("MSE:", mean_squared_error(y_test, y_pred_svm))
print("R2:", r2_score(y_test, y_pred_svm))
print()

# ---- Logistic Regression ----
print("=== Iris – Logistic Regression ===")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("MSE:", mean_squared_error(y_test, y_pred_log))
print("R2:", r2_score(y_test, y_pred_log))
print()


# ---------------------------------
# 2) Supervised Learning – Digits Dataset
# ---------------------------------

print("\n==============================")
print(" 2) Supervised Learning – Digits")
print("==============================\n")

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.3, random_state=42
)

print("=== Digits – KNN Classifier ===")
digits_knn = KNeighborsClassifier(n_neighbors=3)
digits_knn.fit(X_train_d, y_train_d)
y_pred_digits = digits_knn.predict(X_test_d)

print("Accuracy:", accuracy_score(y_test_d, y_pred_digits))
print("MSE:", mean_squared_error(y_test_d, y_pred_digits))
print("R2:", r2_score(y_test_d, y_pred_digits))
print()

# عرض مثال لصورة من مجموعـة Digits
plt.figure()
plt.imshow(digits.images[3], cmap='gray')
plt.title(f"Digits Sample - Label: {digits.target[3]}")
plt.axis('off')
plt.show()


# ---------------------------------
# 3) MNIST Sample Visualization
# ---------------------------------

print("\n==============================")
print(" 3) MNIST Sample Visualization")
print("==============================\n")

(X_train_m, y_train_m), (X_test_m, y_test_m) = mnist.load_data()

plt.figure()
plt.imshow(X_train_m[4], cmap='gray')
plt.title(f"MNIST Sample - Label: {y_train_m[4]}")
plt.axis('off')
plt.show()


# ---------------------------------
# 4) Unsupervised Learning – K-Means (Mall Customers)
# ---------------------------------

print("\n==============================")
print(" 4) Unsupervised Learning – K-Means")
print("==============================\n")

# تأكدي أن ملف Mall_Customers.csv موجود في نفس المجلد
mall = pd.read_csv("Mall_Customers.csv")

# نستخدم عمودين: الدخل السنوي ودرجة الإنفاق
X_mall = mall[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------- Elbow Method --------
inertia = []
K_RANGE = range(1, 10)

for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_mall)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(K_RANGE, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# نختار k = 5 (مثلاً) بناءً على الكوع
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
mall['cluster'] = kmeans.fit_predict(X_mall)

# مراكز المجموعات
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=['Annual Income (k$)', 'Spending Score (1-100)']
)
centers['cluster'] = range(k)

income_med = mall['Annual Income (k$)'].median()
score_med = mall['Spending Score (1-100)'].median()

def get_meaning(income, score):
    """
    يعطي تفسير لكل كلستر بناءً على كونه:
    - High / Low Income
    - High / Low Spending
    """
    if income >= income_med and score >= score_med:
        return "High Income – High Spending"
    elif income >= income_med and score < score_med:
        return "High Income – Low Spending"
    elif income < income_med and score >= score_med:
        return "Low Income – High Spending"
    else:
        return "Low Income – Low Spending"

centers['meaning'] = centers.apply(
    lambda row: get_meaning(row['Annual Income (k$)'], row['Spending Score (1-100)']),
    axis=1
)

print("=== Cluster Centers with Meaning ===")
print(centers)
print()

# -------- Plot Clusters with Meaning-Based Legend --------
plt.figure(figsize=(8, 6))

for idx, info in centers.iterrows():
    cluster_points = mall[mall['cluster'] == info['cluster']]
    plt.scatter(
        cluster_points['Annual Income (k$)'],
        cluster_points['Spending Score (1-100)'],
        s=70,
        label=f"Cluster {info['cluster']}: {info['meaning']}"
    )

# مراكز الكلسترات
plt.scatter(
    centers['Annual Income (k$)'],
    centers['Spending Score (1-100)'],
    s=150,
    c='red',
    marker='X',
    label='Centroids'
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (K-Means Clustering)')
plt.legend()
plt.show()
