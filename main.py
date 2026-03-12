from src.data_loader import load_nhanes_data
from src.preprocessing import create_master_table, clean_and_split
from src.models_engine import train_and_evaluate, get_feature_importance

def main():
    print("=== NUTRITION ML PIPELINE START ===")

    # Loading
    print("Loading NHANES data")
    # Loading the three separate datasets from raw folder
    diet, demo, quest = load_nhanes_data()

    if diet is not None and demo is not None and quest is not None:
        # Preprocessing
        print("Integrating and cleaning data")
        # Merging demographic, dietary, and questionnaire data
        master_df = create_master_table(demo, diet, quest)
        # Handling outliers, imputation, and splitting into train/test sets
        X_train, X_test, y_train, y_test = clean_and_split(master_df)

        # Modeling
        print("Training and evaluating balanced models")
        # Running the optimized suite: Balanced LR, Balanced RF, and XGBoost (scale_pos_weight)
        comparison_table, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)

        print("\n--- FINAL PERFORMANCE METRICS ---")
        print(comparison_table.to_string())

        # XAI (Explainable AI)
        print("\n Extracting Feature Importance (XGBoost)")
        # Extracting feature influence based on the best performing model
        importance_df = get_feature_importance(trained_models["XGBoost"], X_train.columns)

        print("\n--- XAI: TOP RISK FACTORS IDENTIFIED ---")
        print(importance_df.head(5))

        print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
    else:
        print("[ERROR] Data not found. Please check your data/raw/ directory.")

if __name__ == "__main__":
    main()