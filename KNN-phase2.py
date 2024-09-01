
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('cardio_train.csv', delimiter=';')
    df.drop(['id'], axis=1, inplace=True)
    df['age'] = [int(age / 365) for age in df['age']]

    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X = df[features]
    y = df['cardio']

    X = pd.DataFrame(StandardScaler().fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    k_values = range(1, 31)
    metrics = ['euclidean', 'manhattan', 'minkowski']

    best_k_values = {}
    best_accuracy_scores = {}
    best_confusion_matrices = {}

    for metric in metrics:
        best_accuracy = 0
        best_k = 0
        best_cm = None
        accuracy_scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            accuracy_scores.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_cm = cm

            #print(f"Testing k={k} with metric {metric}. Number of test cases executed: {k}/{len(k_values)}")

        best_k_values[metric] = best_k
        best_accuracy_scores[metric] = best_accuracy
        best_confusion_matrices[metric] = best_cm

        plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', label=f'Metric: {metric}')

    plt.figure(figsize=(12, 8))
    plt.title('Accuracy vs. Number of Neighbors (k) for Different Metrics')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Best k values for different metrics:", best_k_values)
    print("Best accuracy scores for different metrics:", best_accuracy_scores)
    print("Best confusion matrices for different metrics:")
    for metric, cm in best_confusion_matrices.items():
        print(f"Metric: {metric}")
        print(cm)


if __name__ == "__main__":
    main()
