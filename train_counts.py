"""Train every count market and save count_model.json."""
import warnings
warnings.filterwarnings("ignore")

from count_model import train_and_save

if __name__ == "__main__":
    print("Fitting count markets...")
    out = train_and_save()
    print(f"\nSaved count_model.json "
          f"({len(out['markets'])} markets, {len(out['referees'])} referees)")
