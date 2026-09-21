"""Train every count market for all five leagues -> count_model.json."""
import warnings
warnings.filterwarnings("ignore")

from count_model import train_and_save
from leagues import LEAGUES

if __name__ == "__main__":
    print("Fitting count markets for the top five leagues...\n")
    out = train_and_save()
    print("\nSaved count_model.json")
    for code, lg in out["leagues"].items():
        print(f"  {LEAGUES[code]['name']:<16} {lg['n_matches']:>5} matches, "
              f"{len(lg['teams']):>2} teams, {len(lg['referees']):>3} referees, "
              f"through {lg['last_date']}")
