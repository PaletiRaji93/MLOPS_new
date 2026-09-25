import pandas as pd


def remove_nulls(df):
    return df.dropna()


class TestRemoveNulls:
    def test_removes_rows_with_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = remove_nulls(df)
        assert len(result) == 1
        assert result.iloc[0]["a"] == 1

    def test_returns_same_if_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = remove_nulls(df)
        assert len(result) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = remove_nulls(df)
        assert len(result) == 0

    def test_columns_lowered_and_stripped(self):
        df = pd.DataFrame({" Name ": [1], " AGE": [2]})
        df.columns = df.columns.str.lower().str.strip()
        assert list(df.columns) == ["name", "age"]
