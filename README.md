# Titanic Survival Prediction

A classification model estimating passenger survival probability aboard the RMS Titanic, built from the 891-passenger training dataset, using demographic, ticket, and travel-group data.

## Business framing

Given a passenger's class, fare, age, sex, and travel details, can we predict whether they survived? Beyond the prediction itself, the more useful question is: **what actually determined survival on the Titanic, and by how much?** This project answers both.

## Dataset

- 891 passengers, 12 original columns
- Target variable: `Survived` (0 = did not survive, 1 = survived)
- Baseline class split: 61.6% did not survive, 38.4% survived — a mild imbalance, accounted for during evaluation and train-test splitting

## Data cleaning

| Column | Issue | Approach |
|---|---|---|
| `Age` | 19.9% missing | Filled using the median age within each Pclass + Sex group, rather than a single overall average, to preserve real age differences between fare classes and genders |
| `Cabin` | 77.1% missing | Too sparse to impute reliably. Extracted the deck letter where available and labelled the rest `Unknown`, treating the presence/absence of a recorded cabin as a feature in itself |
| `Embarked` | 0.2% missing | Filled with the most common boarding port (mode) — negligible impact given only 2 affected passengers |

## Feature engineering

- **`FamilySize`** — combined `SibSp` and `Parch` (+1 for the passenger) into a single travel-group size
- **`IsAlone`** — binary flag derived from `FamilySize`, isolating solo travellers
- **`Deck`** — first letter of `Cabin`, or `Unknown` where absent
- **`Title`** — extracted from `Name` (e.g., "Braund, Mr. Owen Harris" → `Mr`) using pattern matching, then consolidated: French-language variants merged into their English equivalents (`Mlle`, `Ms` → `Miss`; `Mme` → `Mrs`), and ten low-frequency titles (`Dr`, `Rev`, `Col`, `Major`, `Don`, `Lady`, `Sir`, `Capt`, `Countess`, `Jonkheer`) grouped into a single `Rare` category to avoid single-passenger classes with no learnable pattern

## Key findings

**Sex and class compounded rather than acted independently.** Survival ranged from 96.8% (1st class women) down to 13.5% (3rd class men) — an 83-point gap between the two extremes of the same disaster. Notably, a 3rd class woman (50.0%) still outlived a 1st class man (36.9%), showing gender outweighed wealth on its own.

**Title revealed a pattern that sex alone couldn't.** Splitting `Mr` from `Master` — both labelled "male" in the raw data — showed boys survived at 57.5% versus 15.7% for adult men, a more than 3x gap invisible until the title was extracted from the name field. This was measurable evidence for "children first," not just anecdote.

**Family size had a non-linear relationship with survival**, rising from 30.4% (travelling alone) to a peak of 72.4% at a family size of 4, before collapsing to 13–20% for families of 5 or more — consistent with larger groups struggling to stay together or fit into lifeboats collectively. Groups above size 7 had very few passengers each and are noted as low-confidence data points rather than firm trends.

**Whether a cabin was recorded at all was itself predictive.** Passengers with a known cabin survived at roughly double the rate (59–76% depending on deck) of those with no cabin on record (30.0%) — cabin data was missing disproportionately among lower-fare passengers, so its absence functioned as a proxy for class even before the deck letter was considered.

## Modeling

Two models were trained on an 80/20 stratified train-test split (stratified to preserve the 62/38 survival split in both sets) and compared directly:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | **82.1%** | 78.8% |
| Precision (Survived) | 0.79 | 0.74 |
| Recall (Survived) | 0.72 | 0.70 |
| F1 (Survived) | 0.76 | 0.72 |

Logistic Regression, the simpler model, outperformed Random Forest on every metric. On a dataset this size (712 training rows) with already well-engineered features, the added complexity of Random Forest offered no advantage and likely fit some noise in the training data. This was treated as the headline modeling takeaway: **benchmark against a simple baseline before assuming a more complex model will win.**

### Confusion matrix — Logistic Regression (test set, n=179)

| | Predicted: Died | Predicted: Survived |
|---|---|---|
| **Actual: Died** | 97 | 13 |
| **Actual: Survived** | 19 | 50 |

The model's main weakness: recall on survivors (72%) trails its other scores, meaning roughly 1 in 4 actual survivors in the test set were misclassified as non-survivors. Flagged here rather than smoothed over, since this is the metric that would matter most if this were a live safety application rather than a historical dataset.

### Feature importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Fare | 22.0% |
| 2 | Age | 20.6% |
| 3 | Title: Mr | 14.2% |
| 4 | Sex | 10.2% |
| 5 | Pclass | 7.0% |
| 6 | Family Size | 6.2% |
| 7 | Deck: Unknown | 3.7% |

`Fare` and `Age`, both continuous variables, rank above the binary `Sex` column — a known characteristic of tree-based importance scores, which tend to favor features with more possible split points regardless of how informative a binary feature actually is. `Title: Mr` ranking above raw `Sex` (10.2%) confirms the engineered title feature carried more signal than gender alone.

## Testing the model on new, unseen passengers

Beyond scoring the held-out test set, the trained model was used to score two hypothetical passengers built entirely from scratch, to confirm its logic held up on genuinely new input:

- A 1st class woman, age 28, high fare, travelling alone → **94.7% predicted survival probability**
- A 3rd class man, age 40, low fare, travelling alone → **95.0% predicted probability of not surviving**

Both results align directly with the patterns identified during analysis, without either case having existed anywhere in the training data.

## Limitations

- Trained on 891 rows only — a small dataset by modern standards, which likely explains why the simpler model outperformed Random Forest
- No hyperparameter tuning was performed on either model; a tuned Random Forest may close or reverse the current performance gap
- A single train-test split was used rather than cross-validation, so the reported accuracy is a single estimate rather than an averaged, more robust one
- Several engineered categories (e.g., family sizes above 7, deck `T`) are based on very few passengers and should not be treated as reliable trends

## Repository structure

```
titanic-survival-prediction/
├── data/
│   └── Titanic-Dataset.csv
├── notebooks/
│   └── titanic_survival_prediction.ipynb
├── images/
│   ├── survival_by_class_sex.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

*Dataset: Titanic - Machine Learning from Disaster (Kaggle).*
