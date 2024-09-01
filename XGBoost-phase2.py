
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


def read_data():
    path = 'cardio_train.csv'
    df = pd.read_csv(path,delimiter=";")
    x1=df.loc[:,df.columns!='cardio']
    y1=df.loc[:,'cardio']
    return x1, y1


def conf_matrix(ytest, predict):
    cm = confusion_matrix(ytest, predict)
    plt.figure(figsize=(6, 4))
    fg = sns.heatmap(cm, annot=True, cmap="Blues", fmt='d') 
    figure = fg.get_figure()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Output Confusion Matrix")
    return cm


def do_training(x1, y1, training_size=0.10):
    xtrain,xtest,ytrain,ytest=train_test_split(x1, y1, train_size=training_size, random_state=5)
    model=XGBClassifier()
    model.fit(xtrain,ytrain)
    predict=model.predict(xtest)
    accuracy = round(accuracy_score(ytest,predict)*100,3)
    return accuracy, ytest, predict


def main():
    x1, y1 = read_data()
    columns = ["Training Size", "Accuracy"]
    scores_df = pd.DataFrame(columns=columns)
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))

    for ts in range(1, 20, 1):
        ts_rounded = round(ts * 0.05,2)
        acc, ytest, predict = do_training(x1, y1, ts_rounded)
        scores_df = pd.concat([scores_df, pd.DataFrame({"Training Size": [ts_rounded], "Accuracy": [acc]})], ignore_index=True)

        cm = confusion_matrix(ytest, predict)
        ax = axes[ts // 4, ts % 4]
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"Training Size: {ts_rounded}, Accuracy: {acc}%")
        
    plt.tight_layout()
    plt.show()
    print(scores_df)
    return


if __name__ == '__main__':
    main()
