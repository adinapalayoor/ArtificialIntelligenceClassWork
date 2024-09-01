
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC
from imblearn.over_sampling import SMOTE
from sklearn import ensemble
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


def read_data():
    path = 'cardio_train.csv'
    df = pd.read_csv(path,delimiter=";")
    x1=df.loc[:,df.columns!='cardio']
    y1=df.loc[:,'cardio']
    # Resample the training set to balance the classes
    oversampler = SMOTE(random_state=30)
    X_train_resampled, Y_train_resampled = oversampler.fit_resample(x1, y1)
    return X_train_resampled, Y_train_resampled


# def conf_matrix(ytest, predict):
#     cm = confusion_matrix(ytest, predict)
#     plt.figure(figsize=(6, 4))
#     fg = sns.heatmap(cm, annot=True, cmap="Blues", fmt='d') 
#     figure = fg.get_figure()
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title("Output Confusion Matrix")
#     return cm


# def do_training(X_train_resampled, Y_train_resampled, training_size=0.10):
#     xtrain,xtest,ytrain,ytest=train_test_split(X_train_resampled, Y_train_resampled, train_size=training_size, random_state=5)
#     model=XGBClassifier()
#     model.fit(xtrain,ytrain)
#     predict=model.predict(xtest)
#     accuracy = round(accuracy_score(ytest,predict)*100,3)
#     return accuracy, ytest, predict

# def train_ann(X_train_resampled, Y_train_resampled, training_size=0.10):
#     xtrain,xtest,ytrain,ytest=train_test_split(X_train_resampled, Y_train_resampled, train_size=training_size, random_state=5)
#     model = Sequential()
#     model.add(Dense(units=32, activation='relu', input_dim=X_train_resampled.shape[1]))
#     model.add(Dense(units=16, activation='relu'))
#     model.add(Dense(units=8, activation='relu'))
#     model.add(Dense(units=1, activation='sigmoid'))
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', 
#               metrics=[Precision(), Recall(), AUC(), 'accuracy'])
    
#     model.fit(xtrain, ytrain, epochs=10, batch_size=32)
#     predict=model.predict(xtest)
#     accuracy = round(accuracy_score(ytest,predict)*100,3)
    
#     return accuracy, ytest, predict

def train_rf(X_train_resampled, Y_train_resampled, training_size=0.10):
    xtrain,xtest,ytrain,ytest=train_test_split(X_train_resampled, Y_train_resampled, train_size=training_size, random_state=5)
    model = ensemble.RandomForestClassifier()
    model.fit(X_train_resampled, np.ravel(Y_train_resampled))
    y_pred_rfc = model.predict(xtest)
    accuracy = round(accuracy_score(ytest,y_pred_rfc)*100,3)
    return accuracy, ytest, y_pred_rfc


def main():
    X_train_resampled, Y_train_resampled = read_data()
    columns = ["Training Size", "Accuracy"]
    scores_df = pd.DataFrame(columns=columns)

    for ts in range(1, 20, 1):
        ts_rounded = round(ts * 0.05,2)
        print(ts)
        acc_rf, ytest_rf, predict_rf = train_rf(X_train_resampled, Y_train_resampled, ts_rounded)
        scores_df = pd.concat([scores_df, pd.DataFrame({"Training Size": [ts_rounded], "Accuracy": [acc_rf]})], ignore_index=True)
        print(scores_df)

    return


if __name__ == '__main__':
    main()