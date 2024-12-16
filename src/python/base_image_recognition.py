import os
import numpy as np
from icecream import ic
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from Vector_Manager import Vector_Manager


# initialize pretrained model
model = ResNet50(weights='imagenet', include_top=False)


class Image_Manager:
    def __init__(self): 
        self.vector_manager = Vector_Manager(save_path="src/python/")
        self.ipca = IncrementalPCA(n_components=128, batch_size=100)

    # method: to extract image features
    def extract_features(self, image_path, model):
        img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
        
    # function: to save image features
    def save_features(self, features_dict):
        self.vector_manager.save(features_dict)

    # function: to load image features
    def load_features(self):
        return self.vector_manager.load()

    # function: to add a new image 
    def add_image(self, image_path, model, threshold=0.7):
        
        image_name = os.path.basename(image_path)
        
        # load features
        features_dict = self.load_features()
        ic(features_dict)
        features = self.extract_features(image_path, model)
        print(len(features))
        ic(features)
        self.vector_manager.query(vector=features)

        # check, if a similar image exists
        for existing_image, existing_features in features_dict.items():
            similarity = cosine_similarity([features], [existing_features])[0][0]
            
            if similarity > threshold:
                print(f"Warnung: Das Bild '{image_name}' ist ähnlich zu '{existing_image}' (Ähnlichkeit: {similarity:.2f}).")
                return features_dict
        
        # Kein ähnliches Bild gefunden, Merkmale speichern
        features_dict[image_name] = features
        self.save_features(features_dict)
        
        self.vector_manager.add(vector=features)

        print(f"Bild '{image_name}' wurde hinzugefügt.")
        return features_dict


if __name__ == "__main__":
    
    image_manager = Image_Manager()

    # directory containing new images
    image_directory = "src/python/img"
    for image_file in os.listdir(image_directory):
        ic(image_file)
        image_path = os.path.join(image_directory, image_file)
        if os.path.isfile(image_path):
            features_dict = image_manager.add_image(
                image_path=image_path, model=model, threshold=0.3
            )

