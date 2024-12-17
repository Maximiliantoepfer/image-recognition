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
        self.save_path="src/python/"
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
        self.vector_manager.close()
        self.save()


    # method: to extract image features
    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features.flatten()


    # function: to add a new image 
    def add_image(self, image_path, threshold=0.3):
        print("\n")
        image_name = os.path.basename(image_path)
        print(image_name)

        features = self.extract_features(image_path)
        (ids, similarities) = self.vector_manager.search(query_vector=features)
        if ids and similarities:
            for i, id in enumerate(ids): 
                sim = similarities[i]
                if not sim == -1 and sim > threshold:
                    print(f"Warnung: Das Bild '{image_name}' ist zu {(round(sim*100, 2))}% ähnlich zu '{self.id_name_map[id]}'.")
                    return 1

        self.vector_manager.add(id=self.counter, vector=features)
        self.id_name_map[self.counter] = image_name
        self.save()
        self.counter += 1

        print(f"Bild '{image_name}' wurde hinzugefügt.")
        return 0