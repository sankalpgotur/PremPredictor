"""Train the model and save model.json + features.json. Run this to (re)train."""
from epl_predictor import load_data, build_features, evaluate, train_and_save

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} matches across {data['Season'].nunique()} seasons.\n")
    X, y, feature_cols = build_features(data)
    print(f"Built {X.shape[1]} features for {X.shape[0]} matches.\n")
    evaluate(X, y)              # prints accuracy + saves confusion_matrix.png
    train_and_save(X, y, feature_cols)   # refits on ALL data, writes model.json
