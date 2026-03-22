import pickle


class GestureModel:
    def __init__(self, model_path="models/model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, landmarks):
        if len(landmarks) != 63:
            return None
        
        prediction = self.model.predict([landmarks])
        return prediction[0]