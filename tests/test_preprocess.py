from pathlib import Path

import pandas as pd

from churn.data import clean_dataframe, build_preprocessor


def test_clean_dataframe_and_preprocessor(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "tenure": [1, 2, 3, 4],
            "TotalCharges": ["10", "20 ", " ", "40"],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "Churn": ["Yes", "No", "No", "Yes"],
        }
    )

    cleaned = clean_dataframe(df)
    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].isin([0, 1]).all()
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])  # coerced

    pre = build_preprocessor(cleaned)
    X = cleaned.drop(columns=["Churn"])  # type: ignore
    Xt = pre.fit_transform(X)
    # Expect at least as many rows as input
    assert Xt.shape[0] == 4


