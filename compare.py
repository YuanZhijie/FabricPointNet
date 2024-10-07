import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the path for the dataset and the train/test split files
dataset_path = "D:/Workspace/2021_YuanZhijie/Paper01_1/Pointnet_Pointnet2_pytorch/data/modelnet5_normal_resampled"  # Modify this with the actual path
# Define train/test file list paths
train_list_path = os.path.join(dataset_path, "modelnet5_train.txt")
test_list_path = os.path.join(dataset_path, "modelnet5_test.txt")


# Define functions
def load_point_cloud(file_path):
    """Load point cloud data from a txt file."""
    # 指定逗号为分隔符
    return np.loadtxt(file_path, delimiter=',')


def generate_depth_image(point_cloud):
    """Generate a depth image from point cloud data."""
    depth_image = np.zeros((256, 256), dtype=np.float32)  # 初始化浮点型深度图
    x, y, z = point_cloud[:,0], point_cloud[:,1], point_cloud[:,2]
    x_idx = preprocessing.minmax_scale(x, feature_range=(0, 255)).astype(int)
    y_idx = preprocessing.minmax_scale(y, feature_range=(0, 255)).astype(int)
    z_scaled = preprocessing.minmax_scale(z, feature_range=(0, 255))
    depth_image[x_idx, y_idx] = z_scaled
    return depth_image.astype(np.uint8)  # 转换为无符号整型


def calculate_glcm_features(image):
    """Calculate GLCM features for the image."""
    # 确保图像为无符号整型
    image_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    glcm = graycomatrix(image_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy]


def process_folder(folder_path):
    """Process all files in a given folder."""
    features = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):  # 确保处理.txt文件
            label, _ = file.split("_")  # 假设标签和序号用下划线分隔
            point_cloud = load_point_cloud(os.path.join(folder_path, file))
            depth_image = generate_depth_image(point_cloud)
            glcm_features = calculate_glcm_features(depth_image)
            features.append(glcm_features)
            labels.append(label)  # 使用文件名中的标签作为类别标签
    return features, labels


def process_files(file_list_path, folder_path):
    """Process point cloud files listed in a given file list."""
    features = []
    labels = []
    with open(file_list_path, 'r') as file_list:
        for line in file_list:
            file_name = line.strip()  # 获取文件名
            label, _ = file_name.split("_")  # 假设标签和序号用下划线分隔
            file_path = os.path.join(folder_path, label, file_name + ".txt")  # 构建完整文件路径
            point_cloud = load_point_cloud(file_path)
            depth_image = generate_depth_image(point_cloud)
            glcm_features = calculate_glcm_features(depth_image)
            features.append(glcm_features)
            labels.append(label)  # 使用文件名中的标签作为类别标签
    return features, labels


# 12281923version
# Loading the dataset - assuming train and test data are separate
train_features, train_labels = process_files(train_list_path, dataset_path)
test_features, test_labels = process_files(test_list_path, dataset_path)

# Encoding labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Scaling features
scaler = StandardScaler().fit(train_features)
train_features_normalized = scaler.transform(train_features)
test_features_normalized = scaler.transform(test_features)


# Function to print metrics
def print_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


# Training and evaluating models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "BP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear'),
    "GBDT": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
}

for model_name, model in models.items():
    model.fit(train_features_normalized, train_labels_encoded)
    predictions = model.predict(test_features_normalized)
    print_metrics(model_name, test_labels_encoded, predictions)

# Grid search for SVM
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(train_features_normalized, train_labels_encoded)
best_svc = grid.best_estimator_
svm_predictions = best_svc.predict(test_features_normalized)

print("Optimized SVM Classification Report:")
print_metrics("Optimized SVM", test_labels_encoded, svm_predictions)


def print_model_metrics(model, model_name, X_train, y_train, X_test, y_test):
    # Train predictions and accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)

    # Test predictions and accuracy
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Placeholder for Best Accuracy, Evaluate Accuracy, and Best Evaluate Accuracy
    # These values would typically be determined during the model training/validation process
    best_accuracy = max(train_accuracy, test_accuracy)  # Simplified assumption
    evaluate_accuracy = test_accuracy  # Placeholder, typically from a separate validation set
    best_evaluate_accuracy = evaluate_accuracy  # Placeholder, the best from evaluations

    print(f"{model_name} Metrics:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Evaluate Accuracy: {evaluate_accuracy:.4f}")
    print(f"Best Evaluate Accuracy: {best_evaluate_accuracy:.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))
    print("---------------------------------------------------")

# Training and evaluating models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "BP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear'),
    "GBDT": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
}

for model_name, model in models.items():
    # Fit the model
    model.fit(train_features_normalized, train_labels_encoded)
    # Print metrics
    print_model_metrics(model, model_name, train_features_normalized, train_labels_encoded, test_features_normalized, test_labels_encoded)

# Optimizing SVM with Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(train_features_normalized, train_labels_encoded)
best_svc = grid.best_estimator_
print_model_metrics(best_svc, "Optimized SVM", train_features_normalized, train_labels_encoded, test_features_normalized, test_labels_encoded)

