import pickle
model = pickle.load(open("models/best_model.pkl","rb"))
print(model, type(model))
