import pandas as pd
import duckdb
from model import train_era_specific_models


def print_and_save(df: pd.DataFrame, name: str):
    print("\n========================================")
    print(f" FEATURE IMPORTANCE: {name}")
    print("========================================")
    print(df.head(30).to_string(index=False))
    df.to_csv(f"{name}_feature_importance.csv", index=False)
    print(f"Saved: {name}_feature_importance.csv")


def extract_single_model_importance(model, name):
    """Handle a single LightGBM model (runs_class, wicket_flag)."""
    importance = model.feature_importance(importance_type="gain")
    features = model.feature_name()

    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print_and_save(df, name)


def extract_quantile_importance(models_dict, base_name):
    """Handle quantile models like runs_next_over (q5, q50, q95)."""
    for quantile_name, model in models_dict.items():
        importance = model.feature_importance(importance_type="gain")
        features = model.feature_name()

        df = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False)

        full_name = f"{base_name}_{quantile_name}"
        print_and_save(df, full_name)


if __name__ == "__main__":
    print("\n==============================")
    print(" LOADING DATA FROM DUCKDB...")
    print("==============================")

    conn = duckdb.connect("IPL_cricket.db")
    train = conn.execute("SELECT * FROM ml_features_unified_train").fetchdf()
    val = conn.execute("SELECT * FROM ml_features_unified_val").fetchdf()
    conn.close()

    print("Training rows:", len(train))
    print("Validation rows:", len(val))

    print("\n==============================")
    print(" TRAINING MODEL...")
    print("==============================")

    mtm, _ = train_era_specific_models(train, val, modern_only=False)

    print("\n==============================")
    print(" EXTRACTING ALL FEATURE IMPORTANCES")
    print("==============================")

    # 1️⃣ runs_class
    extract_single_model_importance(mtm.models["runs_class"], "runs_class")

    # 2️⃣ wicket_flag
    extract_single_model_importance(mtm.models["wicket_flag"], "wicket_flag")

    # 3️⃣ runs_next_over (q5, q50, q95)
    extract_quantile_importance(mtm.models["runs_next_over"], "runs_next_over")

    # 4️⃣ runs_batsman_next_over (q5, q50, q95)
    extract_quantile_importance(mtm.models["runs_batsman_next_over"], "runs_batsman_next_over")

    print("\n==============================")
    print(" DONE! All feature importance saved.")
    print("==============================")
