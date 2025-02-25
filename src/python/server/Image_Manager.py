import os
import sys
import numpy as np
import pickle
import time
from icecream import ic
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from Vector_Manager import Vector_Manager
import logging

class Image_Manager:
    # Elastic: dimensions=4096, threshold=0.6
    # Faiss: dimensions=100352, threshold=0.3
    def __init__(self, dimensions=0, save_path="data"): 
        self.num_perm=128
        self.save_path=save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.model = ResNet50(weights='imagenet', include_top=False)
        if not dimensions: 
            input_image = np.zeros((1, 224, 224, 3))
            output = self.model.predict(input_image).flatten()
            dimensions = output.shape[0]
        self.vector_manager = Vector_Manager(dimensions=dimensions, save_path=self.save_path)
        self.counter = 0
        self.duration_open_images = 0.0

        self.logger = logging.getLogger(f"image_manager")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler("image_manager.log", mode="w")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)


    def save(self):
        self.vector_manager.save()

    def close(self):
        self.save()
        self.vector_manager.close()
    
    # method: to extract image features
    def extract_features(self, image_path):
        ts = time.time()
        img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
        te = time.time()
        self.duration_open_images += te-ts
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array).flatten()
        return features
    
    # function: to add a new image 
    def add_image(self, image_path :str, id :int):
        # image_name = os.path.basename(image_path)
        if not self.vector_manager.exists_id(id):
            features = self.extract_features(image_path)
            self.vector_manager.add(id=id, vector=features)
            self.counter += 1
            return 0
        else:
            return 1

    def add_bulk_images(self, image_paths :list, ids :list):
        vectors_to_add = []
        ids_to_add = []
        already_existing = []
        try:
            for i, _id in enumerate(ids):
                # image_name = os.path.basename(image_path)
                if self.vector_manager.exists_id(_id):
                    already_existing.append(_id)
                else:
                    features = self.extract_features(image_paths[i])
                    vectors_to_add.append(features)
                    ids_to_add.append(_id)
            if ids_to_add and vectors_to_add:
                self.vector_manager.add_bulk(ids=ids_to_add, vectors=vectors_to_add)
        except Exception as e:
            self.logger.error(f"Error by bulk adding images: {e} \n{sys.exc_info()}")
            return None
        self.logger.info(f"Duration for open images: {self.duration_open_images} sec")
        return {"added": ids_to_add, "already_existing": already_existing}
    
    # function: get similar images 
    def get_similars(self, image_path, threshold=0.3, k=1):
        image_name = os.path.basename(image_path)
        features = self.extract_features(image_path)

        similars = []
        (ids, similarities) = self.vector_manager.search(query_vector=features, k=k)
        for i, id in enumerate(ids): 
            sim = similarities[i]
            if sim > threshold:
                # sim_img_name = self.id_name_map[id]
                self.logger.info(f"Image '{image_name}' is up to {(round(sim*100, 2))}% similar to '{id}'")
                similars.append((i, id, sim))

        return similars


    # delete functions
    def delete_by_ids(self, ids: list):
        return self.vector_manager.delete(ids=ids)

    # def delete_by_name(self, name: str):
    #     id_ = self.vector_manager.get_id_by_name(name)
    #     self.vector_manager.delete(ids=[id_])

