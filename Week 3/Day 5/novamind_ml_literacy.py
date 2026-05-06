"""
NovaMind Classical ML Literacy — Week 3, Day 5
===============================================
PURPOSE: Familiarity only. See the sklearn pattern once so it is not
foreign in interviews or when working alongside data scientists.

This script trains a churn prediction classifier on synthetic NovaMind
customer data. Every line is commented to explain WHAT it does and WHY
it exists — not just what the syntax means.

You will not be tested on implementing this. You need to be able to:
- Read this code and explain what each section does
- Discuss the results in business terms
- Know when to recommend each algorithm to a data scientist colleague

No external services needed. Runs standalone.
Install if needed: pip install scikit-learn
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# ── Step 1: Create synthetic NovaMind customer data ────────────────────────────
# In a real project this comes from a database or CSV.
# We generate it here so the script runs standalone.
#
# Each customer has 5 features:
#   plan_type       : 0=Starter, 1=Professional, 2=Enterprise
#   months_active   : how long they have been a customer
#   monthly_usage   : hours per month they use NovaMind
#   support_tickets : number of support tickets raised (frustration signal)
#   team_size       : number of team members on their account
#
# Label (y):
#   0 = did not churn
#   1 = churned

np.random.seed(42)  # fixed seed so results are reproducible every run
n_customers = 1000  # synthetic dataset size

# Generate feature values with realistic distributions for each plan type
plan_type = np.random.choice([0, 1, 2], size=n_customers, p=[0.6, 0.3, 0.1])
months_active = np.random.randint(1, 36, size=n_customers)
monthly_usage = np.random.uniform(2, 40, size=n_customers)
support_tickets = np.random.poisson(1.5, size=n_customers)
team_size = np.random.randint(1, 30, size=n_customers)

# Build the feature matrix X — shape: (1000 customers, 5 features)
# Each row = one customer, each column = one feature
X = np.column_stack([
    plan_type,
    months_active,
    monthly_usage,
    support_tickets,
    team_size,
])

# Generate churn labels using business logic:
# Higher churn probability when: many support tickets, low usage, short tenure
# This simulates a realistic churn pattern
churn_probability = (
    0.3                                    # base churn rate
    + 0.15 * (support_tickets > 3)         # frustrated customers churn more
    - 0.10 * (monthly_usage > 20)          # engaged customers churn less
    - 0.08 * (months_active > 12)          # long-tenured customers churn less
    + 0.12 * (plan_type == 0)              # Starter plan churns more
    - 0.15 * (plan_type == 2)              # Enterprise churns less
)
churn_probability = np.clip(churn_probability, 0.05, 0.95)
y = np.random.binomial(1, churn_probability)  # 1=churned, 0=stayed

print("=" * 65)
print("  NOVAMIND CHURN PREDICTION — CLASSICAL ML LITERACY")
print("=" * 65)
print(f"\nDataset: {n_customers} customers")
print(f"Features: plan_type, months_active, monthly_usage, support_tickets, team_size")
print(f"Churn rate in dataset: {y.mean():.1%}")
print(f"  ({y.sum()} churned, {(y==0).sum()} stayed)\n")

# ── Step 2: Train/test split ───────────────────────────────────────────────────
# Split data BEFORE training — the test set must never be seen during training.
# test_size=0.2 means 20% held back for honest evaluation.
# random_state fixes the split so results are reproducible.
#
# WHY: if you evaluate on training data, you measure memorisation not learning.
# A model that memorised all 1000 examples scores 100% — meaningless.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y ensures same churn ratio in both train and test sets
    # without this, by random chance test set might have very different churn rate
)

print(f"Training set: {len(X_train)} customers")
print(f"Test set:     {len(X_test)} customers")
print(f"Test churn rate: {y_test.mean():.1%} (should be similar to full dataset)\n")

# ── Step 3: Train three models ─────────────────────────────────────────────────
# We train three algorithms and compare them.
# This mirrors real data science work — never assume one algorithm wins.
#
# model.fit(X_train, y_train) is where learning happens:
#   - Logistic Regression: adjusts weights to find the best linear boundary
#   - Random Forest: builds 100 decision trees on random data subsets
#   - Gradient Boosting: builds trees sequentially, each correcting prior errors
#
# After fit() completes, the model has learned parameters from training data.
# Those parameters are stored inside the model object.

print("Training three models...")

# Logistic Regression — simple linear model, good interpretable baseline
# max_iter=1000 gives it enough iterations to converge
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Random Forest — ensemble of 100 decision trees, robust and powerful
# n_estimators=100 is a hyperparameter YOU set, not something the model learns
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting — sequential tree building, highest accuracy for tabular data
# n_estimators=100 trees, learning_rate controls how much each tree contributes
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

print("All three models trained.\n")

# ── Step 4: Evaluate on test set ───────────────────────────────────────────────
# model.predict() is inference — using learned parameters to predict on NEW data.
# We evaluate on X_test which the model never saw during training.
#
# Four metrics explained:
#   Accuracy  — fraction of all predictions that were correct
#               MISLEADING for imbalanced data (most customers do not churn)
#   Precision — of customers predicted to churn, what fraction actually did?
#               High precision = few false alarms (costly if you send discount to wrong person)
#   Recall    — of customers who actually churned, what fraction did we catch?
#               High recall = catch most churners (costly if you miss a churner)
#   F1 Score  — harmonic mean of precision and recall
#               Use when you need to balance both — single number summary

print("=" * 65)
print("  MODEL COMPARISON — TEST SET RESULTS")
print("=" * 65)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("─" * 65)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}

best_model_name = None
best_f1 = 0

for name, model in models.items():
    # inference — predict on test data the model never saw
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"{name:<25} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\nBest model by F1: {best_model_name}")

# ── Step 5: Cross-validation on best model ────────────────────────────────────
# Cross-validation gives a more reliable performance estimate than a single split.
# cv=5 means 5 folds — model trained and tested 5 times on different splits.
# Every data point is used for testing exactly once.
#
# WHY more reliable: a single 80/20 split might be lucky or unlucky.
# 5-fold averaging smooths out that randomness.

print("\n" + "=" * 65)
print("  CROSS-VALIDATION — RANDOM FOREST (5-fold)")
print("=" * 65)
print("\nWhy cross-validation: single train/test split can be lucky or unlucky.")
print("5-fold uses every data point for testing exactly once.\n")

# Use the full dataset X, y for cross-validation (not just X_train)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="f1")

print(f"F1 scores across 5 folds: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean F1:  {cv_scores.mean():.3f}")
print(f"Std dev:  {cv_scores.std():.3f}  (lower = more consistent across folds)")

# ── Step 6: Feature importance ────────────────────────────────────────────────
# Random Forest can tell you which features mattered most for predictions.
# This is one reason ensemble models are used in business — explainability.
#
# Feature importance = how much each feature reduced prediction error across all trees.
# Higher = more useful for predicting churn.
# This tells the business WHERE to focus — which customer signals matter most.

print("\n" + "=" * 65)
print("  FEATURE IMPORTANCE — RANDOM FOREST")
print("  Which customer signals predict churn most strongly?")
print("=" * 65)

feature_names = ["plan_type", "months_active", "monthly_usage", "support_tickets", "team_size"]
importances = rf_model.feature_importances_

# Sort features by importance descending
sorted_idx = np.argsort(importances)[::-1]

print()
for rank, idx in enumerate(sorted_idx, 1):
    bar = "█" * int(importances[idx] * 50)
    print(f"  {rank}. {feature_names[idx]:<20} {importances[idx]:.3f}  {bar}")

print("\nBusiness interpretation:")
print("  The top feature is what the data science team should investigate first.")
print("  High support_tickets importance → invest in customer success team.")
print("  High plan_type importance → focus retention on Starter plan customers.")

# ── Step 7: Confusion matrix ──────────────────────────────────────────────────
# The confusion matrix shows exactly what the model got right and wrong.
# Four quadrants:
#   True Positive  (TP): predicted churn, actually churned    ← caught churners
#   True Negative  (TN): predicted stayed, actually stayed    ← correct non-churn
#   False Positive (FP): predicted churn, actually stayed     ← false alarm
#   False Negative (FN): predicted stayed, actually churned   ← MISSED churner
#
# For churn prediction: False Negatives are the most costly.
# A missed churner = lost revenue with no intervention attempted.

print("\n" + "=" * 65)
print("  CONFUSION MATRIX — RANDOM FOREST")
print("=" * 65)

rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)

tn, fp, fn, tp = cm.ravel()

print(f"""
                    PREDICTED
                  Stayed  Churned
  ACTUAL  Stayed    {tn:>4}     {fp:>4}    ← {fp} false alarms (sent discount unnecessarily)
          Churned   {fn:>4}     {tp:>4}    ← {fn} MISSED churners (lost revenue)

  True Positives  (caught churners)  : {tp}
  True Negatives  (correct non-churn): {tn}
  False Positives (false alarms)     : {fp}
  False Negatives (missed churners)  : {fn}  ← most costly for NovaMind
""")

# ── Step 8: Business summary ──────────────────────────────────────────────────
# This is how you present ML results to a non-technical stakeholder.
# Numbers without context are meaningless. Always frame in business terms.

print("=" * 65)
print("  BUSINESS SUMMARY")
print("=" * 65)

rf_recall = recall_score(y_test, rf_preds, zero_division=0)
rf_prec = precision_score(y_test, rf_preds, zero_division=0)

print(f"""
  Model: Random Forest Churn Predictor
  Test set: {len(y_test)} customers

  Of every 100 customers who will actually churn:
    → Model correctly identifies {rf_recall*100:.0f} of them (recall)
    → Misses {(1-rf_recall)*100:.0f} without any intervention

  Of every 100 customers the model flags as churning:
    → {rf_prec*100:.0f} will actually churn (precision)
    → {(1-rf_prec)*100:.0f} are false alarms (unnecessary discount offers)

  Recommendation:
    If missing churners is costly (enterprise accounts) → optimise for recall.
    If false alarms are costly (discount budget) → optimise for precision.
    Current model balances both — adjust threshold based on business priority.
""")

print("Lane 1 literacy goal achieved.")
print("You can read this code, explain each section, and discuss results in business terms.")
print("Implementation depth (Lane 2) builds on this foundation.")
