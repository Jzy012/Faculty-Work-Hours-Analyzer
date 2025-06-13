import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from werkzeug.utils import secure_filename
from models import model_utils

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# In-memory storage for latest upload (single user)
uploaded_data = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_data
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            flash("Please upload a valid CSV file.", "danger")
            return redirect(url_for("index"))
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        try:
            df = pd.read_csv(filepath)
            analyzed_df = model_utils.predict_anomalies(df)
            uploaded_data = analyzed_df
            flash("File uploaded and analyzed successfully!", "success")
            return redirect(url_for("faculty_list"))
        except Exception as ex:
            flash(f"Failed to process file: {ex}", "danger")
            return redirect(url_for("index"))
        finally:
            os.remove(filepath)
    return render_template("index.html")

@app.route("/faculty")
def faculty_list():
    global uploaded_data
    if uploaded_data is None:
        flash("No data uploaded yet.", "warning")
        return redirect(url_for("index"))
    faculty_names = sorted(uploaded_data["faculty"].unique())
    return render_template("faculty_list.html", faculty_names=faculty_names)

@app.route("/faculty/<faculty_name>")
def analysis(faculty_name):
    global uploaded_data
    if uploaded_data is None:
        flash("No data uploaded yet.", "warning")
        return redirect(url_for("index"))
    df = uploaded_data[uploaded_data["faculty"] == faculty_name].copy()
    if df.empty:
        flash("No records for the selected faculty.", "warning")
        return redirect(url_for("faculty_list"))
    # Stats
    avg_hours = round(df["work_hours"].mean(), 2)
    anomaly_count = (df["anomaly"] == "Unusual").sum()
    total_records = len(df)
    return render_template(
        "analysis.html",
        faculty=faculty_name,
        records=df.sort_values("date"),
        avg_hours=avg_hours,
        anomaly_count=anomaly_count,
        total_records=total_records,
    )

if __name__ == "__main__":
    # Ensure model exists at startup
    model_utils.load_model()
    app.run(debug=True)