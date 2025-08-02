from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/add', methods=['POST'])
def add():
    try:
        model=joblib.load("Mymodel.h5")
        f=model.feature_names_in_
        d = {
            "trans_date_trans_time": str(request.form['trans_date_trans_time']),
            "cc_num": str(request.form['cc_num']),
            "category": str(request.form['category']),
            "AMT_TRANS": float(request.form['AMT_TRANS']),
            "gender": str(request.form['gender']),
            "street": str(request.form['street']),
            "state": str(request.form['state']),
            "city": str(request.form['city']),
            "dob": str(request.form['dob']),
            "zip": int(request.form['zip']),
            "trans_num": str(request.form['trans_num'])
        }
        d=pd.DataFrame([d])
        d=pd.get_dummies(d)
        d=d.reindex(columns=f,fill_value=0)
        p=model.predict(d)
        print(p)
        if p[0] == 1:
            result="Fraudulent transaction"
        else:
            result="Not Fraudulent transaction"

    except ValueError:
        result = "Invalid input!"
    
    return redirect(url_for('result', result=result))

@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

