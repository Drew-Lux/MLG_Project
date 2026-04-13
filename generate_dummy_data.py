# THis is only used till role 2 has given me a data set. You can delete this once we have that dataset.

import pandas as pd
import numpy as np

# Number of rows
n = 200

rng = np.random.default_rng(42)

df = pd.DataFrame({
    "age": rng.integers(20, 70, size=n),
    "bmi": rng.normal(26, 4, size=n).round(1),
    "blood_glucose": rng.normal(110, 20, size=n).round(1),
    "hba1c": rng.normal(5.8, 0.7, size=n).round(2),
    "exercise_days": rng.integers(0, 7, size=n),
    "diet_score": rng.integers(1, 10, size=n),
    "stress_score": rng.integers(1, 10, size=n),
    "smoking": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    "alcohol": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    "diabetes_stage": rng.choice(
        ["Healthy", "Pre-diabetes", "Diabetes"],
        size=n,
        p=[0.5, 0.3, 0.2]
    )
})

# Save to your data folder
df.to_csv("data/Diabetes_and_Lifestyle_Dataset_Dummy.csv", index=False)
print("Dummy dataset saved → data/Diabetes_and_Lifestyle_Dataset_.csv")