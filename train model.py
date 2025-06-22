import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
df=pd.read_csv("titanic.csv")
df["Sex"]=df["Sex"].map({"male":0,"female":1})
x=df[["Pclass","Sex","Age","Fare"]]
y=df["Survive"]
model=LogisticRegression()
model.fit(x,y)
dump(model,"titanic_model.pkl")
print("model saved successfully!")

