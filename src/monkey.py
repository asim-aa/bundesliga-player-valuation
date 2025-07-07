import joblib
m = joblib.load("models/best_pipeline.pkl")
print(type(m))