import numpy as np
import pandas as pd

INPUT_FILE  = "netflix_titles.csv"
OUTPUT_FILE = "netflix_titles_cleaned.csv"


def load_data(path):
    df = pd.read_csv(path)
    return df


def inspect(df):
    df.info()
    print(df.describe(include="all"))
    print(df.head())
    missing = df.isnull().sum().rename("count").to_frame()
    missing["pct"] = (missing["count"] / len(df) * 100).round(2)
    print(missing[missing["count"] > 0].sort_values("count", ascending=False))


def handle_missing(df):
    df["director"] = df["director"].fillna("Unknown")
    df["cast"]     = df["cast"].fillna("Unknown")
    df["country"]  = df["country"].fillna(df["country"].mode()[0])
    df = df.dropna(subset=["date_added"])

    rating_mode = df["rating"].mode()[0]
    df["rating"] = df["rating"].fillna(rating_mode)

    bad_mask = df["rating"].str.contains("min|Season", na=False)
    df.loc[bad_mask, "duration"] = df.loc[bad_mask, "rating"]
    df.loc[bad_mask, "rating"]   = np.nan
    df["rating"] = df["rating"].fillna(rating_mode)

    return df


def fix_duration(df):
    def _split(raw):
        if pd.isna(raw) or not str(raw).strip():
            return np.nan, np.nan
        parts = str(raw).strip().split()
        try:
            value = int(parts[0])
        except ValueError:
            return np.nan, np.nan
        unit = parts[1].lower() if len(parts) > 1 else ""
        unit = "min" if "min" in unit else ("seasons" if "season" in unit else unit)
        return value, unit

    df[["duration_value", "duration_unit"]] = pd.DataFrame(
        df["duration"].apply(_split).tolist(), index=df.index
    )
    df["duration_value"] = pd.to_numeric(df["duration_value"], errors="coerce")
    return df


def parse_dates(df):
    df["date_added"]  = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    df["year_added"]  = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    return df


def tidy(df):
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    df = df.drop_duplicates()
    return df


def main():
    df = load_data(INPUT_FILE)
    inspect(df)
    df = handle_missing(df)
    df = fix_duration(df)
    df = parse_dates(df)
    df = tidy(df)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
