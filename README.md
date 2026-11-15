# Student Grievance Understanding, Sensitivity Detection and Routing Assistant

**Problem Statement:** LPU-GRIEVANCE-01 | **Course Outcomes:** CO2, CO3, CO4

Academic prototype for student grievance intake, department suggestion, sensitivity triage, and staff queue management.

## Features
- Student grievance submission, optional anonymous submission, and tracking ID lookup.
- Department suggestion using TF-IDF + Logistic Regression.
- Independent phrase-pattern sensitivity checks for Safety, Harassment, and Discrimination.
- Sensitive matches receive Critical priority independently of department-classifier confidence.
- Staff login, priority-ordered grievance queue, case assignment, and status updates.
- SQLite persistence and a basic dashboard.

## Important safety and scope statement
This is a classroom prototype, not an approved LPU production service. The included 38-row training dataset is synthetic. The sensitivity detector is a rule-based triage aid and can miss indirect language or flag benign wording. The classifier has not been evaluated on an independent, representative, human-labelled test set. Model confidence is not a probability that a grievance is true. A trained and authorised human must review every case. Do not enter real student personal or sensitive information into this demonstration.

## Requirements
- Python 3.10 or newer
- Packages in `requirements.txt`
- Internet access only for the optional Chart.js dashboard chart loaded from a CDN

## Run on Windows PowerShell
From the extracted project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

PowerShell activation is not required. Open `http://127.0.0.1:5000`.

## Demonstration staff login
- Username: `admin`
- Password: `ChangeMe123!`

These are local demonstration credentials only. Before any shared or hosted deployment, set unique values using environment variables and implement production security controls. Example PowerShell configuration:

```powershell
$env:ADMIN_USERNAME="your-admin"
$env:ADMIN_PASSWORD="use-a-unique-long-password"
$env:SECRET_KEY="generate-a-long-random-secret"
.\.venv\Scripts\python.exe app.py
```

The database is created as `grievances.db` when the application starts. The trained model is cached at `data/grievance_model.joblib`; remove that file to retrain after changing the CSV.

## Dataset
`data/sample_grievances.csv` has 38 synthetic rows and two columns: `text` and `department`. Labels: Academics (5), Hostel Administration (5), Accounts (5), IT Support (5), Transport (5), Infrastructure (5), Student Welfare (5), and Other (3). It is a demonstration dataset, not an external or real institutional dataset.

## Model and routing
1. TF-IDF creates unigram and bigram text features.
2. Logistic Regression suggests one department and returns a class probability shown as confidence.
3. An independent rule engine checks Safety, Harassment, and Discrimination patterns.
4. A sensitivity match overrides the predicted category and sets priority to Critical regardless of classifier confidence.
5. The staff queue sorts sensitive records first, then Critical, High, Medium, and Low priority.

## Tests
Install requirements, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The included suite has 23 synthetic rule-level checks (15 sensitive positive examples and 8 ordinary/negation examples). The recorded run completed with **23 passed**. This is not a real-world accuracy score and does not establish precision, recall, F1, or generalisation.

## Acceptance criteria — targets to evaluate, not claimed results
- Defined sensitivity regression examples produce the expected sensitivity category.
- Every rule-flagged case receives Critical priority regardless of classifier confidence.
- Every sensitive record appears before every non-sensitive record in the staff queue.
- Each accepted grievance receives a unique tracking ID and can be retrieved.
- Invalid description lengths are rejected (minimum 15 characters; maximum 5,000 characters).
- Staff-only pages require an authenticated staff session.

For independent evaluation, create a privacy-protected, human-labelled holdout set. Report department-classification precision, recall, macro-F1 and a confusion matrix; report sensitivity recall and false-positive rate by category. Record the dataset version, split method, random seed, and software versions.

## Architecture and routes
Browser → Flask → input validation → (TF-IDF/Logistic Regression department suggestion + independent sensitivity rules) → priority override → SQLite → student tracking and staff dashboard.

Main routes: `/`, `/submitted/<tracking_id>`, `/track`, `/admin/login`, `/admin`, `/admin/grievance/<id>`, `/admin/logout`, and `/health`.

## Database
SQLite table `grievances` stores the tracking ID, student fields, anonymity flag, selected department/category, description, predicted department/category, confidence, sensitivity flag/type, priority, status, assigned officer/team, and timestamps.

## Security limitations and deployment warning
The default application entry point is configured for local development and debug mode. It uses a demonstration secret and a single demo administrator. Before deployment: disable debug mode; use HTTPS; configure strong secrets and unique credentials; add CSRF protection, rate limiting, secure cookie settings, role-based permissions, audit logging, encrypted backups, retention/deletion rules, monitoring, and incident response. Obtain institutional approval before processing real complaints. Do not deploy this prototype as a real grievance channel without a security and privacy review.

## Submission placeholders
Complete `PROJECT_LINKS.txt` with the real group registration numbers, a verified GitHub repository URL, and a verified deployed URL. `localhost` is not a live deployment link. Add actual member contributions and genuine screenshots to the report before submission.


## Academic submission notes
This project is a prototype. Synthetic tests do not establish real-world performance. See the IEEE term paper for observed tests, limitations, acceptance criteria and deployment status.
