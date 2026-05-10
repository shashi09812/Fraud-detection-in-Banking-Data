import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D as Convolution2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import pickle
import os

accuracy = []
precision = []
recall = []
fscore = []

print("Loading and preprocessing dataset...")
dataset = pd.read_csv("Dataset/Blockchain_Fraud.csv")
dataset.fillna(0, inplace=True)
Y = dataset['FLAG'].ravel()
dataset = dataset.values
X = dataset[:, 4:dataset.shape[1]-2]
X = normalize(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Using a smaller subset for faster execution during the testing
X = X[0:5000]
Y = Y[0:5000]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def calculateMetrics(algorithm, p_predict, y_test_real):
    a = accuracy_score(y_test_real, p_predict) * 100
    p = precision_score(y_test_real, p_predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test_real, p_predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test_real, p_predict, average='macro', zero_division=0) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(f"{algorithm} executed successfully.")

# Algorithms execution
print("Running Traditional ML Models...")
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
calculateMetrics("Logistic Regression", lr.predict(X_test), y_test)

mlp = MLPClassifier(max_iter=500)
mlp.fit(X_train, y_train)
calculateMetrics("MLP", mlp.predict(X_test), y_test)

nb = GaussianNB()
nb.fit(X_train, y_train)
calculateMetrics("Naive Bayes", nb.predict(X_test), y_test)

ab = AdaBoostClassifier()
ab.fit(X_train, y_train)
calculateMetrics("AdaBoost", ab.predict(X_test), y_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
calculateMetrics("Decision Tree", dt.predict(X_test), y_test)

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
calculateMetrics("SVM", svm_model.predict(X_test), y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
calculateMetrics("Random Forest", rf.predict(X_test), y_test)

# Deep Neural Network
print("Running Deep Neural Network...")
X_dnn = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
Y_dnn = to_categorical(Y)
X_train_dnn, X_test_dnn, y_train_dnn, y_test_dnn = train_test_split(X_dnn, Y_dnn, test_size=0.2)

classifier = Sequential()
classifier.add(Convolution2D(32, (1, 1), input_shape=(X_train_dnn.shape[1], X_train_dnn.shape[2], X_train_dnn.shape[3]), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1, 1)))
classifier.add(Convolution2D(32, (1, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1, 1)))
classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(Y_dnn.shape[1], activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_dnn, y_train_dnn, batch_size=16, epochs=5, shuffle=True, verbose=0)

predict_dnn = classifier.predict(X_test_dnn)
predict_dnn = np.argmax(predict_dnn, axis=1)
y_test_dnn_real = np.argmax(y_test_dnn, axis=1)
calculateMetrics("Deep Neural Network", predict_dnn, y_test_dnn_real)

# Generating Output Visualization
print("Visualizing results and saving to images...")
output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th></tr>"
algorithms = ['Logistic Regression', 'MLP', 'Naive Bayes', 'AdaBoost', 'Decision Tree', 'SVM', 'Random Forest', 'Deep Neural Network']
for i, name in enumerate(algorithms):
    output+=f"<tr><td>{name}</td><td>{accuracy[i]:.2f}</td><td>{precision[i]:.2f}</td><td>{recall[i]:.2f}</td><td>{fscore[i]:.2f}</td></tr>"
output+="</table></body></html>"

with open("table.html", "w") as f:
    f.write(output)
print("Saved table.html")

df_data = []
for i, name in enumerate(algorithms):
    df_data.extend([
        [name, 'Precision', precision[i]],
        [name, 'Recall', recall[i]],
        [name, 'F1 Score', fscore[i]],
        [name, 'Accuracy', accuracy[i]]
    ])

df = pd.DataFrame(df_data, columns=['Algorithms', 'Parameters', 'Value'])
df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar', figsize=(10,6))
plt.title("Performance Metrics of Fraud Detection Algorithms")
plt.ylabel('Score (out of 100)')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results_graph.png")
print("Saved results_graph.png")
print("All validations completed successfully.")
