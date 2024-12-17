import os
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from icecream import ic


class Vector_Manager:
    def __init__(self, save_path="", dimensions=100352):
        print(save_path)
        self.index_file = os.path.join(save_path, "faiss_index.bin")
        try:
            self.index = faiss.read_index(self.index_file)
        except Exception as exception:
            print(f"\n{exception}:\n\nCreating new Index\n")
            base_index = faiss.IndexFlatIP(dimensions)
            self.index = faiss.IndexIDMap(base_index)

        self.save_file = os.path.join(save_path, "image_features.pkl")
        self.id_map = []

    def save(self):
        faiss.write_index(self.index, self.index_file)

    def close(self):
        self.save()


    def add(self, id, vector):
        vector = normalize(np.array([vector]), norm='l2')
        self.index.add_with_ids(vector, np.array([id]))
        self.id_map.append(id)
        return 0

    def search(self, query_vector, k=1):
        query_vector = normalize(np.array([query_vector]), norm='l2')
        sims, ids = self.index.search(query_vector, k=k)
        sims = sims[0]
        ids = ids[0]
        return (ids, sims)

    
