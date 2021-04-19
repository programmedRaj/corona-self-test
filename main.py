import pickle
from flask import Flask, render_template, request

app = Flask(__name__)


file = open(r"C:\Users\user\Desktop\ccl-miniproject\model.pkl", "rb")
clf = pickle.load(file)
file.close()


@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = float(myDict["fever"])
        age = int(myDict["age"])
        bodyPain = int(myDict["bodyPain"])
        runnyNose = int(myDict["runnyNose"])
        diffBreath = int(myDict["diffBreath"])
        travFor = int(myDict["travFor"])
        inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath, travFor]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        infs = clf.predict([inputFeatures])[0]
        print(infs)
        if infs == 0:
            msg = "Low Risk of Infection"

        if infs == 1:
            msg = "High Risk of Infection"

        return render_template("show.html", inf=infProb * 100, r=msg)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
