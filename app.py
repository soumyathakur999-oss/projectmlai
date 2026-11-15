import os
import re
import sqlite3
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from grievance_engine import analyse_grievance, train_or_load_model

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "grievances.db"
MODEL_PATH = BASE_DIR / "data" / "grievance_model.joblib"
DATA_PATH = BASE_DIR / "data" / "sample_grievances.csv"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "local-demo-secret-change-before-deployment")
app.config["ADMIN_USERNAME"] = os.environ.get("ADMIN_USERNAME", "admin")
app.config["ADMIN_PASSWORD_HASH"] = generate_password_hash(
    os.environ.get("ADMIN_PASSWORD", "ChangeMe123!")
)

DEPARTMENTS = [
    "Academics", "Hostel Administration", "Accounts", "IT Support",
    "Transport", "Infrastructure", "Student Welfare", "Other"
]
CATEGORIES = [
    "Attendance", "Examination", "Faculty", "Hostel", "Fees and Refunds",
    "Scholarship", "Network and ERP", "Transport", "Facilities",
    "Harassment", "Safety", "Discrimination", "Other"
]


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id TEXT UNIQUE NOT NULL,
                student_name TEXT,
                student_email TEXT,
                anonymous INTEGER NOT NULL DEFAULT 0,
                department TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                predicted_department TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                sensitive INTEGER NOT NULL DEFAULT 0,
                sensitivity_type TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Submitted',
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.template_filter("pretty_date")
def pretty_date(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return value


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        if len(description) < 15:
            flash("Please describe the issue in at least 15 characters.", "error")
            return render_template("index.html", departments=DEPARTMENTS)

        if len(description) > 5000:
            flash("Please keep the description under 5,000 characters.", "error")
            return render_template("index.html", departments=DEPARTMENTS)

        anonymous = request.form.get("anonymous") == "on"
        student_name = "" if anonymous else request.form.get("student_name", "").strip()
        student_email = "" if anonymous else request.form.get("student_email", "").strip()

        if not anonymous and (not student_name or not student_email):
            flash("Enter your name and email, or select anonymous submission.", "error")
            return render_template("index.html", departments=DEPARTMENTS)

        claimed_department = request.form.get("department", "Not sure")
        category = request.form.get("category", "Other")
        analysis = analyse_grievance(description, MODEL_PATH, DATA_PATH)

        # Sensitive reports always receive critical priority, independent of model confidence.
        priority = "Critical" if analysis["sensitive"] else analysis["priority"]
        tracking_id = "GRV-" + uuid.uuid4().hex[:8].upper()
        now = datetime.now().isoformat(timespec="seconds")

        db = get_db()
        db.execute("""
            INSERT INTO grievances (
                tracking_id, student_name, student_email, anonymous, department,
                category, description, predicted_department, predicted_category,
                confidence, sensitive, sensitivity_type, priority, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Submitted', ?, ?)
        """, (
            tracking_id, student_name, student_email, int(anonymous),
            claimed_department, category, description,
            analysis["department"], analysis["category"],
            analysis["confidence"], int(analysis["sensitive"]),
            analysis["sensitivity_type"], priority, now, now
        ))
        db.commit()
        return redirect(url_for("submission_success", tracking_id=tracking_id))

    return render_template("index.html", departments=DEPARTMENTS)


@app.route("/submitted/<tracking_id>")
def submission_success(tracking_id):
    row = get_db().execute(
        "SELECT tracking_id, created_at FROM grievances WHERE tracking_id = ?",
        (tracking_id,)
    ).fetchone()
    if row is None:
        abort(404)
    return render_template("submitted.html", grievance=row)


@app.route("/track", methods=["GET", "POST"])
def track():
    grievance = None
    searched = False
    if request.method == "POST":
        searched = True
        tracking_id = request.form.get("tracking_id", "").strip().upper()
        grievance = get_db().execute("""
            SELECT tracking_id, predicted_department, priority, status, created_at, updated_at
            FROM grievances WHERE tracking_id = ?
        """, (tracking_id,)).fetchone()
    return render_template("track.html", grievance=grievance, searched=searched)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if (username == app.config["ADMIN_USERNAME"] and
                check_password_hash(app.config["ADMIN_PASSWORD_HASH"], password)):
            session["admin_logged_in"] = True
            return redirect(request.args.get("next") or url_for("admin_dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()
    rows = db.execute("""
        SELECT * FROM grievances
        ORDER BY sensitive DESC,
          CASE priority WHEN 'Critical' THEN 0 WHEN 'High' THEN 1
                        WHEN 'Medium' THEN 2 ELSE 3 END,
          created_at ASC
    """).fetchall()

    counts = {
        "total": db.execute("SELECT COUNT(*) FROM grievances").fetchone()[0],
        "pending": db.execute("SELECT COUNT(*) FROM grievances WHERE status NOT IN ('Resolved','Closed')").fetchone()[0],
        "resolved": db.execute("SELECT COUNT(*) FROM grievances WHERE status = 'Resolved'").fetchone()[0],
        "sensitive": db.execute("SELECT COUNT(*) FROM grievances WHERE sensitive = 1").fetchone()[0],
    }
    dept_rows = db.execute("""
        SELECT predicted_department AS department, COUNT(*) AS total
        FROM grievances GROUP BY predicted_department ORDER BY total DESC
    """).fetchall()
    status_rows = db.execute("""
        SELECT status, COUNT(*) AS total FROM grievances GROUP BY status
    """).fetchall()
    return render_template(
        "admin_dashboard.html", grievances=rows, counts=counts,
        dept_data=[dict(r) for r in dept_rows],
        status_data=[dict(r) for r in status_rows]
    )


@app.route("/admin/grievance/<int:grievance_id>", methods=["GET", "POST"])
@admin_required
def grievance_detail(grievance_id):
    db = get_db()
    grievance = db.execute("SELECT * FROM grievances WHERE id = ?", (grievance_id,)).fetchone()
    if grievance is None:
        abort(404)

    if request.method == "POST":
        status = request.form.get("status", "Submitted")
        assigned_to = request.form.get("assigned_to", "").strip()
        allowed_statuses = {"Submitted", "Under Review", "Forwarded", "Resolved", "Closed"}
        if status not in allowed_statuses:
            abort(400)
        now = datetime.now().isoformat(timespec="seconds")
        db.execute("""
            UPDATE grievances SET status = ?, assigned_to = ?, updated_at = ?
            WHERE id = ?
        """, (status, assigned_to, now, grievance_id))
        db.commit()
        flash("Grievance record updated.", "success")
        return redirect(url_for("grievance_detail", grievance_id=grievance_id))

    return render_template("grievance_detail.html", grievance=grievance)


@app.route("/health")
def health():
    return {"status": "ok", "service": "student-grievance-assistant"}


if __name__ == "__main__":
    init_db()
    train_or_load_model(MODEL_PATH, DATA_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)
