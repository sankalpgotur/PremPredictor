"""Train the corner model and save corners_model.json. Run to (re)train."""
import warnings
warnings.filterwarnings("ignore")

from corners_model import train_and_save, TARGETS

if __name__ == "__main__":
    out = train_and_save()
    print("Saved corners_model.json")
    for t in TARGETS:
        s = out["targets"][t]
        print(f"  {t}: alpha={s['alpha']:.4f}, {len(s['params'])} coefficients")
