# Titanic Benchmark: Question File

Using the provided `train.csv` and `test.csv`, complete the following tasks:

1. **Load & Inspect**

   - Load both datasets into pandas DataFrames. :contentReference[oaicite:0]{index=0}
   - Print shapes and basic statistics (`.shape`, `.info()`, `.describe()`).

2. **EDA & Missing Data**

   - List columns with nulls and their counts. :contentReference[oaicite:1]{index=1}
   - Compute median Age by (`Sex`, `Pclass`) group.
   - Identify the most frequent port in `Embarked`.

3. **Imputation**

   - Fill missing `Age` using group medians.
   - Fill missing `Fare` in the test set with overall median.
   - Fill missing `Embarked` with its mode.

4. **Feature Engineering**

   - Extract `Title` from `Name` and map rare titles to “Rare.”
   - Create `FamilySize = SibSp + Parch + 1`.
   - Create `IsAlone` flag (1 if FamilySize=1, else 0).
   - Bin `Age` into 5 equal-width categories (`pd.cut`).
   - Bin `Fare` into 4 quantile-based categories (`pd.qcut`). :contentReference[oaicite:2]{index=2}

5. **Encoding**

   - Map `Sex` to {male:0, female:1}.
   - Map `Embarked` to {S:0, C:1, Q:2}.
   - Map `Title` to ordinal codes {Mr:1, Miss:2, Mrs:3, Master:4, Rare:5}.

6. **Modeling**

   - Train **LogisticRegression** (scikit-learn default settings) with 5-fold CV; report mean accuracy. :contentReference[oaicite:3]{index=3}
   - Train **RandomForestClassifier** (100 trees) with 5-fold CV; report mean accuracy. :contentReference[oaicite:4]{index=4}

7. **Evaluation & Submission**

   - Fit the best model on full training data.
   - Predict on test set.
   - Save `submission.csv` containing `PassengerId` and `Survived`.

8. **Optional Extras**
   - Plot feature importances for the Random Forest.
   - Compute ROC AUC on a held-out validation split.
