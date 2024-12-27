import os
import numpy as np
import pickle
from icecream import ic
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Vector_Manager import Vector_Manager


class Image_Manager:
    def __init__(self): 
        self.save_path="src/python/data"
        os.makedirs(self.save_path, exist_ok=True)
        self.vector_manager = Vector_Manager(save_path=self.save_path)
        self.model = ResNet50(weights='imagenet', include_top=False)
        self.counter = 0
        self.id_name_map = {}
        self.id_name_map_file = os.path.join(self.save_path, "id_name_map.pkl")
        if os.path.exists(self.id_name_map_file):
            with open(self.id_name_map_file, "rb") as f:
                self.id_name_map = pickle.load(f)

    def save(self):
        self.vector_manager.save()
        with open(self.id_name_map_file, "wb") as f:
            pickle.dump(self.id_name_map, f)

    def close(self):
        self.save()
        self.vector_manager.close()


    # method: to extract image features
    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features.flatten()

    
    # function: to add a new image 
    def add_image(self, image_path :str, id :int):
        image_name = os.path.basename(image_path)
        features = self.extract_features(image_path)
        if not self.vector_manager.exists_id(id=id):
            self.vector_manager.add(id=id, vector=features)
            self.id_name_map[id] = image_name
            self.save()
            self.counter += 1
            return 0
        else:
            return 1

    
    # function: get similar images 
    def get_similars(self, image_path, threshold=0.3, k=1):
        image_name = os.path.basename(image_path)
        features = self.extract_features(image_path)
        similars = []
        (ids, similarities) = self.vector_manager.search(query_vector=features, k=k)
        ic(ids)
        ic(similarities)
        for i, id in enumerate(ids): 
            sim = similarities[i]
            if sim > threshold:
                ic(self.id_name_map)
                ic(id)
                new_img_name = self.id_name_map[id]
                print(f"Warn: Image '{image_name}' is up to {(round(sim*100, 2))}% similar to '{new_img_name}'")
                similars.append((i, id, new_img_name, sim))
        return similars


    # delete functions
    def delete_by_ids(self, ids: list):
        self.vector_manager.delete(ids=ids)

    def delete_by_name(self, name: str):
        id_ = self.vector_manager.get_id_by_name(name)
        self.vector_manager.delete(ids=[id_])

