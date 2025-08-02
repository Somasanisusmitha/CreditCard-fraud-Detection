import pandas as pd
import joblib

# Load model for prediction
model = joblib.load("Mymodel.h5")
f = model.feature_names_in_

# Input data for prediction
a = input("Enter the trans_date_trans_time:\n")
b = input("Enter ccnum:\n")
g = input("Enter category:\n")
c = float(input("transaction amount:\n"))
x = input("Enter gender:\n")
y_s = input("Enter street:\n")
s = input("Enter state:\n")
t = input("Enter city:\n")
v = input("Enter dob:\n")
z = input("Enter zip :\n")
w = input("Enter job:\n")
y_t = input("enter transaction number:\n")

# Create DataFrame for input data
d = {
    "trans_date_trans_time": a,
    "cc_num": b,
    "category": g,
    "amt_trans": c,
    "gender": x,
    "street": y_s,
    "state": s,
    "city": t,
    "dob": v,
    "zip": z,
    "job": w,
    "trans_num": y_t
}
d = pd.DataFrame([d])
d = pd.get_dummies(d)
d = d.reindex(columns=f, fill_value=0)

# Predict
p = model.predict(d)
print("Prediction:",p)
'''if p[0] == 1:
    print("Fraudulent transaction")
else:
    print("Not Fraudulent transaction")'''
