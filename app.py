from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

filename = 'model3.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        Pclass = request.form['Pclass']
        Sex = request.form['Sex']
        Age = request.form['Age']
        SibSp = request.form['SibSp']
        Parch = request.form['Parch']
        Fare = request.form['Fare']
        result = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]])

        prediction = model.predict(result)
        if prediction == 0:
            display = "Survived in Titanic Incident"
            print(display)
        elif prediction == 1:
            display = "Died in Titanic Incident"
            print(display)

    return render_template("submit.html", n=display)


if __name__ == "__main__":
    app.run(debug = True)



