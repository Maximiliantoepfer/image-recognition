import os
import pickle
import faiss
from icecream import ic

class Vector_Manager:
    def __init__(self, save_path=""):
        print(save_path)
        self.index = faiss.IndexFlatIP(100352)
        self.save_file = os.path.join(save_path, "image_features.pkl")

    def add(self, vector):
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        self.index.add(vector)
        # self.index.add_with_ids
        return 0

    def query(self, query_vector):
        ic(query_vector)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        sim, ids = self.index.search(query_vector, k=5)
        ic(f"Top 5 ähnliche Vektoren (IDs): {ids}")
        ic(f"Cosinus-Ähnlichkeiten: {sim}")


    # function: to save image features
    def save(self, vectores_dict):
        with open(self.save_file, "wb") as f:
            pickle.dump(vectores_dict, f)
        return 0

    # function: to load image features
    def load(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                return pickle.load(f)
        return {}
    
