from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_select_best_model


def main():
    print("Starting training pipeline...")

    df = load_data(DATA_PATH)

    model, best_name, best_mae, validation_results = train_and_select_best_model(df)

    print("\nTraining completed!")
    print(f"Best Model: {best_name}")
    print(f"Test MAE: {best_mae:.2f}")

    print("\nValidation Results:")
    print(f"CV MAE Scores: {validation_results['cv_mae_scores']}")
    print(f"Average CV MAE: {validation_results['cv_mean_mae']}")
    print(f"CV MAE Std Dev: {validation_results['cv_std_mae']}")


if __name__ == "__main__":
    main()