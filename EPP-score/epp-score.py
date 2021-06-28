from EloPy import elopy
i = elopy.Implementation()
i.addPlayer("Hank")
i.addPlayer("Bill")
print(i.getPlayerRating("Hank"))
#----------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/train.csv")
cols_to_keep = ["Survived","Age","Fare"]
dummy_Pclass = pd.get_dummies(df["Pclass"],prefix="Pclass")
dummy_Sex = pd.get_dummies(df["Sex"],prefix="Sex")

data = df[cols_to_keep].join(dummy_Pclass.loc[:,"Pclass_2":])
data = data.join(dummy_Sex.loc[:,"Sex_male":])

X = data[:,1:]
y = data[:,1]

print(data.head())
clf = RandomForestClassifier(max_depth=2)
clf.fit(X,y)

#------------------------------------------





