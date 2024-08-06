from flask import Flask, render_template, request
from pickle import load
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        fn = "dia_lr.pkl"
        if os.path.exists(fn):
            f = open(fn, "rb")
            model = load(f)
            f.close()

            fs = float(request.form["fs"])
            fu = request.form["fu"]

            if fu == "1":
                d = [[fs, 1]]
            else:
                d = [[fs, 0]]

            result = model.predict(d)

            if result == "YES":
                msg = "You have a chance of Diabetes"
            elif result == "NO":
                msg = "You don't have Diabetes"
            else:
                msg = "Something gone wrong"

            return render_template("home.html", msg=msg)
        else:
            print(fn, "does not exist")
            return "Model file not found"

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
