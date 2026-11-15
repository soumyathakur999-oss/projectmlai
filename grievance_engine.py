"""NLP classification and deterministic safety escalation rules.

The classifier suggests a department. Sensitive-case rules are independent of
classifier confidence and always override the predicted priority.
"""
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Patterns deliberately cover common paraphrases, not only exact keywords.
# This is a triage aid, not a substitute for trained human review.
SENSITIVE_PATTERNS = {
    "Harassment": [
        r"\b(harass(?:ed|ment|ing)?|stalk(?:ed|ing)?|bully(?:ing|ied)?|blackmail(?:ed|ing)?)\b",
        r"\b(unwanted|repeated|persistent)\s+(?:messages?|texts?|calls?|contact|comments?|advances?)\b",
        r"\b(?:keeps?|repeatedly)\s+(?:sending|send|messages?|contacting|making)\b.{0,100}\b(unwanted|inappropriate|sexual|personal)\b",
        r"\b(inappropriate|sexual)\s+(?:comments?|messages?|touching|remarks?|advances?)\b",
        r"\basked\s+(?:them|him|her)\s+to\s+stop\b.{0,100}\b(continues?|continued|again|keeps?)\b",
        r"\b(unwanted touching|inappropriate touching|sexual coercion)\b",
    ],
    "Safety": [
        r"\b(i feel unsafe|i am unsafe|not safe|afraid for my safety|fear for my safety)\b",
        r"\b(threat(?:en|ened|ening|ens)?|threats?)\b.{0,100}\b(hurt|harm|attack|kill|physically|safety|me|my)\b",
        r"\b\b(?:hurt|harm|attack|kill)\b.{0,80}\b(me|my|physically|student)\b",
        r"\b(physical threat|physical violence|physical assault|assault|weapon|violence)\b",
        r"\b(following me|someone is following|afraid to return|afraid to go back)\b",
    ],
    "Discrimination": [
        r"\b(discriminat\w*|caste[- ]based|racial abuse|religious abuse|sexist abuse)\b",
        r"\b(treat\w*|judg\w*|exclud\w*|den\w*|target\w*)\b.{0,100}\bbecause of\s+(?:my|their|his|her)\s+(background|caste|religion|race|gender|disability|identity)\b",
        r"\b(unfair treatment)\b.{0,100}\b(background|caste|religion|race|gender|disability)\b",
    ],
}


def detect_sensitivity(text):
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not normalized:
        return False, None
    # Avoid a small set of explicit negations being treated as reports.
    negated = re.search(r"\b(?:no|not|never)\s+(?:being\s+|been\s+)?(?:harassed|threatened|discriminated against)\b", normalized)
    if negated:
        return False, None
    # Safety is checked first because immediate physical danger takes precedence.
    for label in ("Safety", "Harassment", "Discrimination"):
        if any(re.search(pattern, normalized) for pattern in SENSITIVE_PATTERNS[label]):
            return True, label
    return False, None


def train_or_load_model(model_path, data_path):
    model_path = Path(model_path)
    if model_path.exists():
        return joblib.load(model_path)

    data = pd.read_csv(data_path).dropna(subset=["text", "department"])
    data["text"] = data["text"].astype(str)
    data["department"] = data["department"].astype(str)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0))
    ])
    pipeline.fit(data["text"], data["department"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def analyse_grievance(text, model_path, data_path):
    model = train_or_load_model(model_path, data_path)
    probabilities = model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    department = model.classes_[best_index]
    confidence = float(probabilities[best_index])

    sensitive, sensitivity_type = detect_sensitivity(text)
    lowered = str(text).lower()

    category = "General"
    category_rules = [
        (["attendance", "absent", "marked present"], "Attendance"),
        (["exam", "examination", "result", "marks", "paper"], "Examination"),
        (["hostel", "room", "warden", "mess"], "Hostel"),
        (["fee", "refund", "scholarship", "payment"], "Fees and Refunds"),
        (["wifi", "wi-fi", "internet", "erp", "login"], "Network and ERP"),
        (["bus", "transport", "route"], "Transport"),
        (["bullying", "harassment", "stalking", "unwanted messages"], "Harassment"),
        (["unsafe", "threat", "assault", "violence", "hurt me", "harm me"], "Safety"),
        (["discrimination", "caste", "sexist"], "Discrimination"),
    ]
    # The sensitivity type takes precedence over ordinary topic keywords.
    if sensitive:
        category = sensitivity_type
    else:
        for keywords, label in category_rules:
            if any(word in lowered for word in keywords):
                category = label
                break

    if sensitive:
        priority = "Critical"
    elif any(word in lowered for word in ["urgent", "tomorrow", "deadline", "exam today", "cannot access"]):
        priority = "High"
    elif any(word in lowered for word in ["again", "third time", "multiple complaints", "for a week"]):
        priority = "High"
    elif any(word in lowered for word in ["suggestion", "please consider", "would be helpful"]):
        priority = "Low"
    else:
        priority = "Medium"

    return {
        "department": department,
        "category": category,
        "confidence": confidence,
        "sensitive": sensitive,
        "sensitivity_type": sensitivity_type,
        "priority": priority,
    }
