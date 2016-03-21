""" Working skript for the project "Titanic"
Author : Ekaterina Tcareva
Date : 8rd March 2016

Our goal is to build a prediction on surviving Titanic passengers.

We have some information about passengers (train.csv):
PassengerId -- A numerical id assigned to each passenger.
Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
Name -- the name of the passenger.
Sex -- The gender of the passenger -- male or female.
Age -- The age of the passenger. Fractional.
SibSp -- The number of siblings and spouses the passenger had on board.
Parch -- The number of parents and children the passenger had on board.
Ticket -- The ticket number of the passenger.
Fare -- How much the passenger paid for the ticker.
Cabin -- Which cabin the passenger was in.
Embarked -- Where the passenger boarded the Titanic.
"""

import pandas
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

# Data cleanup
# TRAIN DATA
titanic = pandas.read_csv("train.csv")

# Fill NAN 
titanic['Age'] =titanic['Age'].fillna(titanic['Age'].median())

titanic['Parch'] =titanic['Parch'].fillna(titanic['Parch'].median())
titanic['SibSp'] =titanic['SibSp'].fillna(titanic['SibSp'].median())

titanic['Sex'] = titanic['Sex'].fillna('male')

# Replace all the occurences of male with the number 0, female 1.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"]=='S', "Embarked"] = 0
titanic.loc[titanic["Embarked"]=='C', "Embarked"] = 1
titanic.loc[titanic["Embarked"]=='Q', "Embarked"] = 2

#Now we try different models and check which one is the best fit

#We started with Logistic regression using cross validation (3 folders) from “sklearn”:
#we are using almost all features except “name”, “cabin”, and “ticket” since we have too much NA
# for “cabin” and we do not know what the numbers of “ticket” and letters mean


alg =linear_model.LogisticRegression(random_state=1)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print("Logistic reg with cross validation using [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]")
print(scores.mean())
# 0.787878787879

# now let's try same predictors but without cross validation (we use whole dataset to fit the algorithm)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict_proba(titanic[predictors].astype(float))[:,1]
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("Logistic reg WITHOUT cross validation using [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]")
print(accuracy)
# 0.799102132435

#Since 0.799 is bigger than 0.788 we suggest that we have some overfitting. Thus, we can try to use few features.

#Next we wanted to explore which of the features are more correlated with the labels.
# We used  Logistic regression for only one feature with cross validation.

scores = cross_validation.cross_val_score(alg, titanic[["Pclass"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Pclass only")
print(scores.mean())
#0.679012345679

scores = cross_validation.cross_val_score(alg, titanic[["Sex"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Sex only")
print(scores.mean())
#0.786756453423

scores = cross_validation.cross_val_score(alg, titanic[["Age"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Age only")
print(scores.mean())
#0.616161616162

scores = cross_validation.cross_val_score(alg, titanic[["Embarked"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Embarked only")
print(scores.mean())
#0.594837261504

scores = cross_validation.cross_val_score(alg, titanic[["Fare"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Fare only")
print(scores.mean())
#0.672278338945

scores = cross_validation.cross_val_score(alg, titanic[["SibSp"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using SibSp only")
print(scores.mean())
#0.616161616162

scores = cross_validation.cross_val_score(alg, titanic[["Parch"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using Parch only")
print(scores.mean())
#0.607182940516

#We can see that logistic regression with only “Sex” feature gives us almost the same accuracy as with all features!

#“Pclass” and “Fare” gave us almost the same prediction and hence we can suggest that they are very closely related to each other.
# Thus, we decided to check their relation. We tried to predict “Fare” using logistic regression and “Pclass” feature:

scores = cross_validation.cross_val_score(alg, titanic[["Fare"]], titanic["Pclass"], cv=3)
print("Logistic reg to predict Pclass using only Fare feature with cross validation")
print(scores.mean())
#0.693573836191 It was surprising, but they do not have a strong correlation. Hence we could use them both.

#Remember - we had overfitting when we used
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#To avoid overfitting we run logistic regression using only three features:
predictors = ["Pclass", "Sex", "Fare"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using [Pclass, Sex, Fare]")
print(scores.mean())
#0.780022446689

# now let's try same predictors but without cross validation (we use whole dataset to fit the algorithm)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict_proba(titanic[predictors].astype(float))[:,1]
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("Logistic reg WITHOUT cross validation using [Pclass, Sex, Fare]")
print(accuracy)
#0.785634118967 the results with and without cross validation is almost same. Thus, now we avoid overfitting using few features.

#At the beginning of our work we decided to create a new feature “famillySize”. Thus, we are creating it:
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

scores = cross_validation.cross_val_score(alg, titanic[["FamilySize"]], titanic["Survived"], cv=3)
print("Logistic reg with cross validation using FamilySize only")
print(scores.mean())
#0.612794612795  So, this new feature is not very related with our labels.

#Next we tried Naive Bayes since we can suggest that most of the features in the data set are independent
#  (we did not use “Fare” since we will be using “Pclass”):


alg = GaussianNB()

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Naive Bayes with cross validation using [Pclass, Sex, Age, SibSp, Parch, Embarked]")
print(scores.mean())
#0.765432098765

predictors = ["Pclass", "Sex", "Age"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Naive Bayes with cross validation using [Pclass, Sex, Age]")
print(scores.mean())
#0.777777777778 The result is worser than the result we got using Logistic Regression.
# So, we decided to  not explore this model deeper.

#Random Forest
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Random Forest with cross validation using [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize]")
print(scores.mean())
#0.817059483726

alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict_proba(titanic[predictors].astype(float))[:,1]
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("Random Forest WITHOUT cross validation using [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize]")
print(accuracy)
#0.918069584736 It is better than using logistic regression, but it is clear that we have overfitting again.
# Thus, we need to use fewer features.

# Also we think that we need to split the “Age” feature into some groups. For example: 0 -13, 13-17, 18-25, 26-50, 51-80.
# To find the best split we use matplotlib to plot the data:
survivedByAge = titanic.groupby('Age')['Survived'].sum()
numberPasByAge = titanic.groupby('Age')['Survived'].count()
numberDiedByAge = numberPasByAge - survivedByAge

a = numberPasByAge.index.values.tolist()
d = numberDiedByAge.values.tolist()
s = survivedByAge.values.tolist()

# now we can plot d and c vs. age
style.use('ggplot')
x = a
y = s
x2 = a
y2 = d
plt.figure(1)
plt.plot(x,y,'bo', label = 'Survived')
plt.plot(x2,y2,'rx', label ='Died')
plt.legend(loc='upper left')
plt.title('Survived vs. Died')
plt.ylabel('Number of people')
plt.xlabel('Age')
#plt.show()

#Looking at this graph we decided to split the “Age” into three groups: <15, 16-30, >31
#Then we created a new feature “AgeGroup”: 0 for persons  < 16, 1 for persons between 16 and 31,
# and 2 for persons older than 31.

titanic.loc[titanic['Age'] < 16, 'AgeGroup'] = 0
titanic.loc[titanic['Age'] >= 16,'AgeGroup'] = 1
titanic.loc[titanic['Age'] >= 31,'AgeGroup'] = 2

# Now we can try SelectKBest in from to find the best features:
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "AgeGroup"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.figure(2)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()

#From the graph we see that Sex, Pclass, and Fare are the most important features!
#And we realised that “AgeGroup” is less important than age.
#Using this information we tried some combinations of features and got the best result using:
predictors = ["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked"]

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Random Forest with cross validation using [Pclass, Sex, Age, Parch, Fare, Embarked]")
print(scores.mean())
#0.82379349046

#Next, to improve our model, we used GradientBoostingClassifier (we used the random forest with the same set of features).
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2), predictors]]


# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("GradientBoostingClassifier with cross validation using [Pclass, Sex, Age, Parch, Fare, Embarked]")
print(accuracy)
#0.824915824916 The result is slightly better than the result we got by just using Random Forest with the same parameters.
#We submitted this result on Kaggle and it gave us 2392nd/3616 and about 0.777 accuracy.
# We realized we had overfitting again! Hence we decided to try the same algorithms with few features:
predictors = ["Pclass", "Sex", "Fare", "Embarked"]

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("GradientBoostingClassifier with cross validation using [Pclass, Sex, Fare, Embarked]")
print(accuracy)
#0.824915824916

# We submitted submitted.py file which used this algorithm and this set of features.
# We got 891st/3661 (0.79904 accuracy)
#https://www.kaggle.com/etcareva/results


