# ============================================
# Bài tập: Nhận dạng ký tự dùng thuật toán Naïve Bayes
# Sinh viên: ..................................
# Môn: Trí tuệ nhân tạo nâng cao
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# 1. Đọc dữ liệu
# =======================

print("Đọc dữ liệu...")
column_names = ["letter"] + [f"f{i}" for i in range(1, 17)]
data = pd.read_csv("C:/Users/ADMIN/Desktop/Word 3/BT5_Bayes/letter-recognition.data", names=column_names)

print("5 dòng đầu của dữ liệu:")
print(data.head())

# =======================
# 2. Tách feature và label
# =======================

X = data.drop("letter", axis=1)
y = data["letter"]

# =======================
# 3. Chia train và test
# =======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Số mẫu train:", len(X_train))
print("Số mẫu test :", len(X_test))

# =======================
# 4. Huấn luyện mô hình
# =======================

model = GaussianNB()
model.fit(X_train, y_train)

# =======================
# 5. Dự đoán
# =======================

y_pred = model.predict(X_test)

# =======================
# 6. Đánh giá mô hình
# =======================

acc = accuracy_score(y_test, y_pred)
print("\nĐộ chính xác (Accuracy):", acc)
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# =======================
# 7. Vẽ confusion matrix
# =======================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

