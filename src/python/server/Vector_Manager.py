import os
import time
import pickle
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from icecream import ic
import logging

class Vector_Manager:
    def __init__(self, dimensions, save_path="", logger=None):
        self.duration_save_faiss = 0.0
        self.duration_save_pickle = 0.0
        self.duration_add_faiss = 0.0
        self.duration_add_pickle = 0.0
        log_filename = f"vector_manager.log"
        self.logger = logging.getLogger(f"vector_manager")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(save_path)
        self.index_file = os.path.join(save_path, "faiss_index.bin")
        try:
            ts_read_index = time.time()
            self.index = faiss.read_index(self.index_file)
            te_read_index = time.time()
            duration_read_index = round(te_read_index - ts_read_index, 2)
            self.logger.info(f"Duration for reading faiss index: {duration_read_index} sec")
        except Exception as exception:
            self.logger.info(f"\n{exception}:\n\nCreating new Index\n")
            ic(dimensions)
            base_index = faiss.IndexFlatIP(dimensions)
            self.index = faiss.IndexIDMap(base_index)

        # self.save_file = os.path.join(save_path, "image_features.pkl")
        ts_load_ids = time.time()
        self.save_file = os.path.join(save_path, "ids.pkl") 
        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                self.ids = pickle.load(f)
        else: 
            self.ids = []
        te_load_ids = time.time()
        duration_load_ids = round(te_load_ids - ts_load_ids, 2)
        self.logger.info(f"Duration for loading ids: {duration_load_ids} sec")

    def save(self):
        ts_save_index = time.time()
        faiss.write_index(self.index, self.index_file)
        te_save_index = time.time()
        duration_save_index = round(te_save_index - ts_save_index, 2)
        self.logger.info(f"Duration for save faiss index: {duration_save_index} sec")

        ts_save_ids = time.time()
        with open(self.save_file, "wb") as f:
            pickle.dump(self.ids, f)
        te_save_ids = time.time()
        duration_save_ids = round(te_save_ids - ts_save_ids, 2)
        self.logger.info(f"Duration for saving ids: {duration_save_ids} sec")

        self.logger.info(f"Total Duration for saving: {duration_save_ids+duration_save_index} sec")



    def close(self):
        self.save()

    def add_bulk(self, ids, vectors): 
        vectors = normalize(np.array(vectors), norm='l2')
        ids = np.array(ids, dtype=np.int64)

        ts_add = time.time()
        self.index.add_with_ids(vectors, ids)
        te_add = time.time()
        dur1 = round(te_add - ts_add, 2)
        self.logger.info(f"Add to index took: {dur1} sec")

        ts_add = time.time()
        self.ids.extend(ids)
        te_add = time.time()
        dur2 = round(te_add - ts_add, 2)
        self.logger.info(f"Add to ids took: {dur2} sec")

        self.logger.info(f"Add took {dur1+dur2} sec in total")
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
        self.logger.info(self.index.d)  # Gibt die erwartete Dimension des Indexes zurück
        self.logger.info(query_vector.shape)  # Sollte (d,) sein, wobei d die Dimension ist
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
            #         self.logger.info(f"Removing ID {id}")
            self.index.remove_ids(np.array(ids, dtype=np.int64))
            for id_ in ids:
                self.ids.remove(id_)
            self.save()
            self.logger.info(f"IDs {ids} successfully removed.")
        else:
            self.logger.info(f"ID {ids} not found in the index.")

    def exists_id(self, id):
        return id in self.ids