from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_select_best_model


def main():
    print("Starting training pipeline...")

    df = load_data(DATA_PATH)

    model, best_model_name, best_mae = train_and_select_best_model(df)

    print("\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Best test MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()