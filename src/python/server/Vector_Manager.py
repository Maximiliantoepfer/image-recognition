import os
import time
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from icecream import ic


class Vector_Manager:
    def __init__(self, dimensions, save_path=""):
        self.duration_save_faiss = 0.0
        self.duration_save_pickle = 0.0
        self.duration_add_faiss = 0.0
        self.duration_add_pickle = 0.0
        print(save_path)
        self.index_file = os.path.join(save_path, "faiss_index.bin")
        try:
            self.index = faiss.read_index(self.index_file)
        except Exception as exception:
            print(f"\n{exception}:\n\nCreating new Index\n")
            ic(dimensions)
            base_index = faiss.IndexFlatIP(dimensions)
            self.index = faiss.IndexIDMap(base_index)

        # self.save_file = os.path.join(save_path, "image_features.pkl")
        self.save_file = os.path.join(save_path, "ids.pkl") 
        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                self.ids = pickle.load(f)
        else: 
            self.ids = []

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.save_file, "wb") as f:
            pickle.dump(self.ids, f)


    def close(self):
        self.save()

    def add_bulk(self, ids, vectors): 
        vectors = normalize(np.array(vectors), norm='l2')
        ids = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        self.ids.extend(ids)
        self.save()


    def add(self, id, vector):
        vector = normalize(np.array([vector]), norm='l2')
        self.index.add_with_ids(vector, np.array([id]))
        self.ids.append(id)
        self.save()
        # self.id_map.append(id)
        return 0

    def search(self, query_vector, k=1):
        query_vector = normalize(np.array([query_vector]), norm='l2')
        print(self.index.d)  # Gibt die erwartete Dimension des Indexes zurück
        print(query_vector.shape)  # Sollte (d,) sein, wobei d die Dimension ist
        sims, ids = self.index.search(query_vector, k=k)
        sims = sims[0]
        ids = ids[0]
        return (ids, sims)

    # def get_id_by_name(self, id):
    #     if id in self.id_map:
    #         return self.id_map[id]
    #     else:
    #         return None

    def delete(self, ids: list):
        if ids:
            # existing_ids = []
            # for id in ids:
            #     if id in self.id_map:
            #         existing_ids.append(id)
            #         self.id_map.remove(id)
            #         print(f"Removing ID {id}")
            self.index.remove_ids(np.array(ids, dtype=np.int64))
            for id_ in ids:
                self.ids.remove(id_)
            self.save()
            print(f"IDs {ids} successfully removed.")
        else:
            print(f"ID {ids} not found in the index.")

    def exists_id(self, id):
        return id in self.ids