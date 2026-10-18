import pandas as pd
import plotly.express as px


def get_overview(df: pd.DataFrame) -> dict:
    """Basic facts about the dataset."""
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numeric columns": len(df.select_dtypes(include="number").columns),
        "Categorical columns": len(df.select_dtypes(exclude="number").columns),
        "Duplicate rows": int(df.duplicated().sum()),
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Column names, data types and number of unique values."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Unique values": df.nunique().values,
        }
    )


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage for every column."""
    missing = df.isnull().sum()
    out = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2),
        }
    )
    return out.sort_values("Missing", ascending=False).reset_index(drop=True)


def plot_missing(missing_df: pd.DataFrame):
    """Bar chart of columns that have missing values (None if there are none)."""
    data = missing_df[missing_df["Missing"] > 0]
    if data.empty:
        return None
    return px.bar(data, x="Column", y="Missing %", title="Missing values by column")


def plot_correlation(df: pd.DataFrame):
    """Correlation heatmap of numeric columns (None if fewer than 2)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation heatmap",
    )


def plot_distribution(df: pd.DataFrame, column: str):
    """Histogram for numeric columns, bar chart of top values for others."""
    if pd.api.types.is_numeric_dtype(df[column]):
        return px.histogram(
            df, x=column, marginal="box", title=f"Distribution of {column}"
        )
    counts = df[column].value_counts().head(20).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title=f"Top values in {column}")
