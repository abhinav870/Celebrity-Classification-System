from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from Project import X,y,class_dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scaling our Data using Standard Scaler. We are randomly choosing these parameters
# We will later fine-tune these paramenters using gridSearchCV
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

print(classification_report(y_test, pipe.predict(X_test)))

"""
Let's use GridSearch to try out different models with different paramets.
Goal is to come up with best model with best fine tuned parameters
"""

# Defining Different Candidate Models with different parameters in the form of a dictionary as shown below
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
            # Implementing SVM with diff values of C like: 1,10,100,1000
            # Choosing Different Kernels like RBF, Linear
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

"""
Implementing GridSearchCV for fine tuning our parameters
"""

scores = []
best_estimators = {}

# Iterating through each model created above
for algo, mp in model_params.items():

    # Creating pipeline and scaling the data
    pipe = make_pipeline(StandardScaler(), mp['model'])

    # Using the model to train
    # We are doing 5-fold Cross Validation
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)  # Fitting the model

    # Appending the scores in a list after each iteration
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

# At the end create a DataFrame which consists of Performance of each of the models on CV-Set
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

# Seeing the parameters of each of the models which gave the best result
print("Printing the parameters of each Model having best F1 Score ",best_estimators)

# Seeing the Performance of different ML Models on the test  set
print("Evaluating the Test Set Using SVM ML Model ",best_estimators['svm'].score(X_test,y_test))
print("Evaluating the Test Set Using Random Forest ML Model ",best_estimators['random_forest'].score(X_test,y_test))
print("Evaluating the Test Set Using Logistic Regression ML Model ",best_estimators['logistic_regression'].score(X_test,y_test))

# Choosing SVM Model as it had the highest F1 Score on the CV-Set
best_clf = best_estimators['svm']

# Building the Confusion Matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

"""
Save the trained model
"""
# Save the model as a pickle in a file
joblib.dump(best_clf, 'saved_model.pkl')

"""
Save the Class Dictionary
"""

with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))