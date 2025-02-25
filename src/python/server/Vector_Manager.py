import os
import time
import math
import pickle
import glob
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from icecream import ic
import logging

class Vector_Manager:
    def __init__(self, dimensions, save_path=""):
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

        self.dimensions = dimensions
        self.save_path = save_path
        self.logger.info(save_path)
        self.indices = {}
        load_start = time.time()
        self.index_paths = glob.glob(f"{save_path}/faiss_index-*.bin")
        for path in self.index_paths:
            index_name = os.path.splitext(os.path.basename(path))[0]
            self.logger.info(index_name)
            try: 
                self.indices[index_name] = faiss.read_index(path)
            except Exception as e:
                self.logger.error(f"Error by reading faiss index {path}")
        load_end = time.time()
        self.logger.info(f"Duration for loading indices: {round(load_end-load_start, 2)} sec")
        # self.index_file = os.path.join(save_path, "faiss_index.bin")
        # try:
        #     ts_read_index = time.time()
        #     self.index = faiss.read_index(self.index_file)
        #     te_read_index = time.time()
        #     duration_read_index = round(te_read_index - ts_read_index, 2)
        #     self.logger.info(f"Duration for reading faiss index: {duration_read_index} sec")
        # except Exception as exception:
        #     self.logger.info(f"\n{exception}:\n\nCreating new Index\n")
        #     ic(dimensions)
        #     base_index = faiss.IndexFlatIP(dimensions)
        #     self.index = faiss.IndexIDMap(base_index)

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
    
    def load_indices(self, index_names):
        if index_names:
            self.index_paths = [f"{self.save_path}/{name}.bin" for name in index_names]
        else:
            self.index_paths = glob.glob(f"{self.save_path}/faiss_index-*.bin")
        for path in self.index_paths:
            index_name = os.path.splitext(os.path.basename(path))[0]
            self.logger.info(index_name)
            try: 
                self.indices[index_name] = faiss.read_index(path)
            except Exception as e:
                self.logger.error(f"Error by reading faiss index {path}")

    def get_index_file_path(self, index_name): 
        return f"{self.save_path}/{index_name}.bin"
    
    def create_faiss_index(self, index_file, index_name):
        self.logger.info(f"Creating new Index {index_name}")
        base_index = faiss.IndexFlatIP(self.dimensions)
        self.indices[index_name] = faiss.IndexIDMap(base_index)
        faiss.write_index(self.indices[index_name], index_file)

    def save_index(self, index_file, index_name):
        ts_save_index = time.time()
        faiss.write_index(self.indices[index_name], index_file)
        te_save_index = time.time()
        duration_save_index = round(te_save_index - ts_save_index, 2)
        self.logger.info(f"Duration for save faiss index: {duration_save_index} sec")
        
    def save_ids(self):
        ts_save_ids = time.time()
        with open(self.save_file, "wb") as f:
            pickle.dump(self.ids, f)
        te_save_ids = time.time()
        duration_save_ids = round(te_save_ids - ts_save_ids, 2)
        self.logger.info(f"Duration for saving ids: {duration_save_ids} sec")


    # def save(self):
    #     ts_save_index = time.time()
    #     faiss.write_index(self.index, self.index_file)
    #     te_save_index = time.time()
    #     duration_save_index = round(te_save_index - ts_save_index, 2)
    #     self.logger.info(f"Duration for save faiss index: {duration_save_index} sec")

    #     ts_save_ids = time.time()
    #     with open(self.save_file, "wb") as f:
    #         pickle.dump(self.ids, f)
    #     te_save_ids = time.time()
    #     duration_save_ids = round(te_save_ids - ts_save_ids, 2)
    #     self.logger.info(f"Duration for saving ids: {duration_save_ids} sec")

    #     self.logger.info(f"Total Duration for saving: {duration_save_ids+duration_save_index} sec")

    def save(self):
        for index in self.indices.keys():
            self.save_index(self.get_index_file_path(index), index)
        self.save_ids()

    def close(self):
        self.save()
        self.indices.clear()

    def add_bulk(self, ids, vectors): 
        id_groups = {}
        for i, _id in enumerate(ids): 
            group_name = math.ceil(_id/10000)*10000
            if not group_name in id_groups.keys():
                id_groups[group_name] = {}
                id_groups[group_name]["ids"] = [_id,]
                id_groups[group_name]["vectors"] = [vectors[i],]
            else:
                id_groups[group_name]["ids"].append(_id)
                id_groups[group_name]["vectors"].append(vectors[i]) # TODO für alle so implementieren
        
        for group_name in id_groups.keys():
            index_name = f"faiss_index-{group_name}"
            
            index_file = self.get_index_file_path(index_name=index_name)
            if not index_name in self.indices.keys():
                self.create_faiss_index(index_file=index_file, index_name=index_name)
            
            _vectors = normalize(np.array(id_groups[group_name]["vectors"]), norm='l2')
            _ids = np.array(id_groups[group_name]["ids"], dtype=np.int64)
            self.indices[index_name].add_with_ids(_vectors, _ids)
            self.ids.extend(ids)
        self.save()
        id_groups.clear()

        # vectors = normalize(np.array(vectors), norm='l2')
        # ids = np.array(ids, dtype=np.int64)

        # ts_add = time.time()
        # self.index.add_with_ids(vectors, ids)
        # te_add = time.time()
        # dur1 = round(te_add - ts_add, 2)
        # self.logger.info(f"Add to index took: {dur1} sec")

        # ts_add = time.time()
        # self.ids.extend(ids)
        # te_add = time.time()
        # dur2 = round(te_add - ts_add, 2)
        # self.logger.info(f"Add to ids took: {dur2} sec")

        # self.logger.info(f"Add took {dur1+dur2} sec in total")
        # self.save()


    def add(self, id, vector):
        id_group = math.ceil(id/10000)*10000
        index_name = f"faiss_index-{id_group}"

        index_file = self.get_index_file_path(index_name=index_name)
        if not index_name in self.indices.keys():
            self.create_faiss_index(index_file=index_file, index_name=index_name)
        
        _vectors = normalize(np.array([vector]), norm='l2')
        _ids = np.array([id], dtype=np.int64)
        self.indices[index_name].add_with_ids(_vectors, _ids)
        self.ids.append(id)
        self.save(index_file, index_name)

        # vector = normalize(np.array([vector]), norm='l2')
        # self.index.add_with_ids(vector, np.array([id]))
        # self.ids.append(id)
        # self.save()
        # self.id_map.append(id)
        return 0

    def search(self, query_vector, k=1):
        query_vector = normalize(np.array([query_vector]), norm='l2')
        # self.logger.info(self.index.d)  # Gibt die erwartete Dimension des Indexes zurück
        # self.logger.info(query_vector.shape)  # Sollte (d,) sein, wobei d die Dimension ist
        # sims, ids = self.index.search(query_vector, k=k)
        # sims = sims[0]
        # ids = ids[0]
        ids = []
        sims = []
        print(self.indices.keys())
        for index in self.indices.keys():
            _sims, _ids = self.indices[index].search(query_vector, k=k)
            sims.extend(_sims[0])
            ids.extend(_ids[0])
        return (ids[0:k+1], sims[0:k+1])

    # def get_id_by_name(self, id):
    #     if id in self.id_map:
    #         return self.id_map[id]
    #     else:
    #         return None

    def delete(self, ids: list):
        if ids:
            all_removed_ids = []
            self.ids = list(dict.fromkeys(self.ids))
            id_groups = {}
            for _id in ids: 
                id_group = math.ceil(_id/10000)*10000
                if not id_group in id_groups.keys():
                    id_groups[id_group] = [_id,]
                else:
                    id_groups[id_group].append(_id)
            for group_name in id_groups.keys():
                ids = id_groups[group_name]
                index_name = f"faiss_index-{group_name}"
                if not index_name in self.indices.keys():
                    continue
                ids_to_remove = []
                for id_ in ids:
                    if id_ in self.ids:
                        ic(id_ in self.ids)
                        ids_to_remove.append(id_)
                        self.ids.remove(id_)
                if ids_to_remove:
                    self.indices[index_name].remove_ids(np.array(ids_to_remove, dtype=np.int64))
                all_removed_ids.extend(ids_to_remove)
            self.save()
            id_groups.clear()
            return all_removed_ids
        return []
        # if ids:
        #     self.index.remove_ids(np.array(ids, dtype=np.int64))
        #     for id_ in ids:
        #         self.ids.remove(id_)
        #     self.save()
        #     self.logger.info(f"IDs {ids} successfully removed.")
        # else:
        #     self.logger.info(f"ID {ids} not found in the index.")

    def exists_id(self, id):
        return id in self.ids
    
