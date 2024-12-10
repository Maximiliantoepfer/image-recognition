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
import pickle

import ElasticImageManager

# initialize pretrained model
model = ResNet50(weights='imagenet', include_top=False)
elastic_manager = ElasticImageManager

# function: to extract image features
def extract_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# function: to save image features
def save_features(features_dict, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(features_dict, f)

# function: to load image features
def load_features(save_path):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return {}

# function: to add a new image 
def add_image(image_path, features_dict, model, save_path, threshold=0.7):
    new_features = extract_features(image_path, model)
    image_name = os.path.basename(image_path)

    # Training der PCA basierend auf bestehenden Features
    pca = PCA(n_components=512)
    if features_dict:
        existing_features = np.array(list(features_dict.values()))
        pca.fit(existing_features)  # Trainiere PCA auf bestehenden Daten
    
    # Reshape und Reduktion
    new_features_reshaped = new_features.reshape(1, -1)
    reduced_vector = pca.transform(new_features_reshaped).flatten()

    ic(reduced_vector)  # Debugging
    ic(image_name, len(reduced_vector))


    # check, if a similar image exists
    for existing_image, existing_features in features_dict.items():
        similarity = cosine_similarity([new_features], [existing_features])[0][0]
        # elastic_similarity = elastic_manager.search_similar_images(query_vector= new_features)
        
        if similarity > threshold:
            print(f"Warnung: Das Bild '{image_name}' ist ähnlich zu '{existing_image}' (Ähnlichkeit: {similarity:.2f}).")
            return features_dict  # Abbruch ohne Hinzufügen
    
    # Kein ähnliches Bild gefunden, Merkmale speichern
    features_dict[image_name] = new_features
    save_features(features_dict, save_path)
    # elastic_manager.index_image(image_name=image_name, image_vector=new_features)
    print(f"Bild '{image_name}' wurde hinzugefügt.")
    return features_dict

if __name__ == "__main__":
    # path to feature file
    features_file = "src/python/image_features.pkl"
    
    # load features
    features_dict = load_features(features_file)

    # directory containing new images
    image_directory = "src/python/img"
    for image_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_file)
        if os.path.isfile(image_path):
            features_dict = add_image(
                image_path=image_path, features_dict=features_dict, model=model, 
                save_path=features_file, threshold=0.3
                )

