import os
import time
import math
import pickle
import glob
import faiss
import numpy as np
import multiprocessing
import gc
from sklearn.preprocessing import normalize
from icecream import ic
import logging

INDEX_SIZE = 2000  # prefer 2000
if __name__ != "__main__":
    multiprocessing.set_start_method('fork', force=True)

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
        self.logger.info("Vector_Manager is starting ...")

        self.dimensions = dimensions
        self.save_path = save_path
        self.logger.info(save_path)
        self.indices = {}
        load_start = time.time()
        # self.load_indices()
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
    
    def get_index_file_path(self, index_name): 
        return f"{self.save_path}/{index_name}.bin"

    def get_all_available_index_names(self):
        index_names = []
        paths = glob.glob(f"{self.save_path}/faiss_index-*.bin")
        for path in paths:
            index_names.append(os.path.splitext(os.path.basename(path))[0])
        return index_names

    def load_indices(self, index_names=[]):
        if index_names:
            paths = [self.get_index_file_path(name) for name in index_names]
        else:
            paths = glob.glob(f"{self.save_path}/faiss_index-*.bin")
        for path in paths:
            index_name = os.path.splitext(os.path.basename(path))[0]
            try: 
                s = time.time()
                if os.path.isfile(path):
                    self.indices[index_name] = faiss.read_index(path)
                else: 
                    self.create_faiss_index(index_file=path, index_name=index_name)
                e = time.time()
                dur = round(e-s, 2)
                self.logger.info(f"{index_name} took {dur} sec")
            except Exception as e:
                self.logger.error(f"Error by reading faiss index {path} - {e}")
                if index_name in self.indices.keys():
                    self.indices.pop(index_name)
                self.create_faiss_index(index_file=path, index_name=index_name)
                self.logger.info(f"Created new faiss index {index_name} for {path}")

    def create_faiss_index(self, index_file, index_name):
        self.logger.info(f"Creating new Index {index_name}")
        base_index = faiss.IndexFlatIP(self.dimensions)
        self.indices[index_name] = faiss.IndexIDMap(base_index)
        faiss.write_index(self.indices[index_name], index_file)

    def save_index(self, index_file, index_name):
        faiss.write_index(self.indices[index_name], index_file)
  
    def save_ids(self):
        with open(self.save_file, "wb") as f:
            pickle.dump(self.ids, f)

    def save(self):
        ts_save_index = time.time()
        if self.indices:
            for index in self.indices.keys():
                self.save_index(self.get_index_file_path(index), index)
        te_save_index = time.time()
        duration_save_indices = round(te_save_index - ts_save_index, 2)
        self.logger.info(f"Duration for save faiss indices: {duration_save_indices} sec")
      
        ts_save_ids = time.time()
        self.save_ids()
        te_save_ids = time.time()
        duration_save_ids = round(te_save_ids - ts_save_ids, 2)
        self.logger.info(f"Duration for saving ids: {duration_save_ids} sec")

    def close(self):
        self.save()
        self.indices.clear()
        gc.collect()

    def add_bulk(self, ids, vectors): 
        id_groups = {}
        for i, _id in enumerate(ids): 
            group_name = math.ceil(_id/INDEX_SIZE)*INDEX_SIZE
            if not group_name in id_groups.keys():
                id_groups[group_name] = {}
                id_groups[group_name]["ids"] = [_id,]
                id_groups[group_name]["vectors"] = [vectors[i],]
            else:
                id_groups[group_name]["ids"].append(_id)
                id_groups[group_name]["vectors"].append(vectors[i]) 
        index_names = [f"faiss_index-{group_name}" for group_name in id_groups.keys()]
        self.load_indices(index_names=index_names)
        for group_name in id_groups.keys():
            index_name = f"faiss_index-{group_name}"
            _vectors = normalize(np.array(id_groups[group_name]["vectors"]), norm='l2')
            _ids = np.array(id_groups[group_name]["ids"], dtype=np.int64)
            self.indices[index_name].add_with_ids(_vectors, _ids)
            self.ids.extend(ids)
        self.close()
        id_groups.clear()
        return 0


    def add(self, id, vector):
        id_group = math.ceil(id/INDEX_SIZE)*INDEX_SIZE
        index_name = f"faiss_index-{id_group}"
        self.load_indices(index_names=[index_name])
        
        _vectors = normalize(np.array([vector]), norm='l2')
        _ids = np.array([id], dtype=np.int64)
        self.indices[index_name].add_with_ids(_vectors, _ids)
        self.ids.append(id)
        self.close()
        return 0

    def search_in_index(self, args):
        index_name, norm_vec, k = args
        index_path = self.get_index_file_path(index_name=index_name)
        try:
            index = faiss.read_index(index_path)
            result = index.search(norm_vec, k=k)
            del index
            return result
        except Exception as e:
            self.logger.error(f"Exception while reading and searching index {index_path}: {e}")
            return ()
        
    def search(self, query_vector, k=1):
        norm_vec = normalize(np.array([query_vector]), norm='l2')
        ids = []
        sims = []
        self.logger.info("Start search")
        start = time.time()
        index_names = self.get_all_available_index_names()
        if not index_names:
            self.logger.warn("No indices to search on")
            return None
        self.logger.info("Start search 2")
        args = [(index_name, norm_vec, k) for index_name in index_names]

        self.logger.info("Start search 3")
        pool = multiprocessing.Pool(processes=10)
        try:
            results = pool.map(self.search_in_index, args)
        finally:
            pool.close()
            pool.join()

        self.logger.info("Search finished")
        for _sims, _ids in results:
            ids.extend(_ids[0])
            sims.extend(_sims[0])

        end = time.time()
        dur = round(end-start, 2)
        self.logger.info(f"Searching took {dur} sec")

        # for index in self.indices.keys():
        #     start = time.time()
        #     _sims, _ids = self.indices[index].search(norm_vec, k=k)
        #     sims.extend(_sims[0])
        #     ids.extend(_ids[0])
        #     end = time.time()
        #     dur = round(end-start, 2)
        #     self.logger.info(f"Searching through Index {index} took {dur} sec")
        return (ids[0:k+1], sims[0:k+1])

    def delete(self, ids: list):
        available_index_names = self.get_all_available_index_names()
            
        if ids and available_index_names:
            all_removed_ids = []
            self.ids = list(dict.fromkeys(self.ids))
            id_groups = {}
            for _id in ids: 
                id_group = math.ceil(_id/INDEX_SIZE)*INDEX_SIZE
                if not id_group in id_groups.keys():
                    id_groups[id_group] = [_id,]
                else:
                    id_groups[id_group].append(_id)

            groups_to_load = [group_name for group_name in id_groups.keys() if f"faiss_index-{group_name}" in available_index_names]
            indices_to_load = [f"faiss_index-{group_name}" for group_name in groups_to_load]
            self.load_indices(index_names=indices_to_load)
            for group_name in groups_to_load:
                ids = id_groups[group_name]
                index_name = f"faiss_index-{group_name}"
                ids_to_remove = []
                for id_ in ids:
                    if id_ in self.ids:
                        ids_to_remove.append(id_)
                        self.ids.remove(id_)
                if ids_to_remove:
                    self.indices[index_name].remove_ids(np.array(ids_to_remove, dtype=np.int64))
                all_removed_ids.extend(ids_to_remove)
            self.close()
            id_groups.clear()
            return all_removed_ids
        gc.collect()
        return []

    def exists_id(self, id):
        return id in self.ids
    
