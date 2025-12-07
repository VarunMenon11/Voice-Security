import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC

X = np.load("X_spoof.npy")
y = np.load("y_spoof.npy")

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(Xtrain, ytrain)

pred = clf.predict(Xtest)

print(classification_report(ytest, pred))

# save model
import joblib
joblib.dump(clf, "spoof_rf.pkl")


svm = SVC(kernel='linear', probability=True)
svm.fit(Xtrain, ytrain)
pred_svm = svm.predict(Xtest)
print(classification_report(ytest, pred_svm))