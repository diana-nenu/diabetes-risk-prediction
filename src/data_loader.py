import pandas as pd
import os


def load_nhanes_data():
    """
    Loads NHANES files from the local directory and returns them as DataFrames.
    """
    base_path = "data/raw/nhanes/"

    try:
        # Loading the three main files you downloaded from Kaggle
        diet = pd.read_csv(os.path.join(base_path, "diet.csv"))
        demo = pd.read_csv(os.path.join(base_path, "demographic.csv"))
        quest = pd.read_csv(os.path.join(base_path, "questionnaire.csv"))

        print(f"Successfully loaded NHANES: {len(diet)} diet records, {len(demo)} demographics.")
        return diet, demo, quest
    except FileNotFoundError as e:
        print(f"Error: Could not find NHANES files. Check your paths. {e}")
        return None, None, None


def load_usda_data():
    """
    Loads USDA Foundation Foods data.
    """
    base_path = "data/raw/usda/"
    try:
        food = pd.read_csv(os.path.join(base_path, "food.csv"), low_memory=False)
        # Added low_memory=False to fix the DtypeWarning
        nutrients = pd.read_csv(os.path.join(base_path, "food_nutrient.csv"), low_memory=False)
        print(f"Successfully loaded USDA: {len(food)} food items.")
        return food, nutrients
    except Exception as e:
        print(f"Error loading USDA data: {e}")
        return None, None