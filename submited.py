""" Writing my first randomforest code.
Author : Ekaterina Tcareva
Date : 8rd March 2016
""" 
import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


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

#create a new feature sexFamily:
# single man - 1, man with SibSp - 2, man with Parch - 3,
# single women - 4, women with Parch - 6, women with SibSp - 5, kids under 16 - 7

titanic.loc[titanic['Age'] < 16, 'sexFamily'] = 7
titanic.loc[(titanic['Age'] > 15.5) & (titanic['Sex'] == 1)  , 'sexFamily'] = 4
titanic.loc[(titanic['Age'] > 15.5) & (titanic['Sex'] == 0)  , 'sexFamily'] = 1
titanic.loc[(titanic['SibSp'] != 0) & (titanic['Sex'] == 1) & (titanic['Age'] > 15.5), 'sexFamily'] = 5
titanic.loc[(titanic['Parch'] != 0) & (titanic['Sex'] == 1) & (titanic['Age'] > 15.5), 'sexFamily'] = 6
titanic.loc[(titanic['SibSp'] != 0) & (titanic['Sex'] == 0) & (titanic['Age'] > 15.5), 'sexFamily'] = 2
titanic.loc[(titanic['Parch'] != 0) & (titanic['Sex'] == 0) & (titanic['Age'] > 15.5), 'sexFamily'] = 3

# TEST DATA
test = pandas.read_csv('test.csv')

# Fill NAN 
test['Age'] =test['Age'].fillna(test['Age'].median())

test['Parch'] =test['Parch'].fillna(test['Parch'].median())
test['SibSp'] =test['SibSp'].fillna(test['SibSp'].median())
test['Cabin'] =test['Cabin'].fillna(0)
test['Ticket'] =test['Ticket'].fillna(0)
test['Pclass'] =test['Pclass'].fillna(test['Pclass'].median())
test['Fare'] =test['Fare'].fillna(test['Fare'].median())


test['Sex'] = test['Sex'].fillna('male')

# Replace all the occurences of male with the number 0, female 1.
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna('S')
test.loc[test["Embarked"]=='S', "Embarked"] = 0
test.loc[test["Embarked"]=='C', "Embarked"] = 1
test.loc[test["Embarked"]=='Q', "Embarked"] = 2

#create a new feature sexFamily:
# single man - 1, man with SibSp - 2, man with Parch - 3,
# single women - 4, women with Parch - 6, women with SibSp - 5, kids under 16 - 7

test.loc[test['Age'] < 16, 'sexFamily'] = 7
test.loc[(test['Age'] > 15.5) & (test['Sex'] == 1)  , 'sexFamily'] = 4
test.loc[(test['Age'] > 15.5) & (test['Sex'] == 0)  , 'sexFamily'] = 1
test.loc[(test['SibSp'] != 0) & (test['Sex'] == 1) & (test['Age'] > 15.5), 'sexFamily'] = 5
test.loc[(test['Parch'] != 0) & (test['Sex'] == 1) & (test['Age'] > 15.5), 'sexFamily'] = 6
test.loc[(test['SibSp'] != 0) & (test['Sex'] == 0) & (test['Age'] > 15.5), 'sexFamily'] = 2
test.loc[(test['Parch'] != 0) & (test['Sex'] == 0) & (test['Age'] > 15.5), 'sexFamily'] = 3
 
predictors = ["Pclass", "Sex", "Fare", "Embarked"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2), predictors]]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv("etcareva2.csv", index=False)
