from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_select_best_model
from src.validate import run_cross_validation


def main():
    print("Starting training pipeline...")

    df = load_data(DATA_PATH)

    _, best_model_name, best_mae = train_and_select_best_model(df)

    print("\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Best test MAE: {best_mae:.2f}")

    print("\nRunning 5-fold cross-validation...")
    cv_scores, avg_cv_mae = run_cross_validation(df)

    print("Cross-validation MAE scores:", cv_scores)
    print(f"Average CV MAE: {avg_cv_mae:.2f}")
    
if __name__ == "__main__":
    main()    