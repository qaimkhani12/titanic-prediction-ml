from joblib import load
model=load("titanic_model.pkl")
result=model.predict([[3,0,25,7.5]])
print("prediction:",result[0])