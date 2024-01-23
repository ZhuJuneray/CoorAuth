from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from ml4auth import knn_multi_kfolds


def generate_test_data(samples=10, features=20, classes=2):
    # 生成随机分类数据集
    data, labels = make_classification(n_samples=samples, n_features=features, n_classes=classes, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, labels

# 调用生成测试数据的函数
data_scaled, labels = generate_test_data()

# 调用您的 knn_multi_kfolds 函数
knn_multi_kfolds(n_neighbors=3, n_splits=5, data_scaled=data_scaled, labels=labels)
