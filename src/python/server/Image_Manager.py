import os
import numpy as np
import pickle
from icecream import ic
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from Vector_Manager import Vector_Manager
from datasketch import MinHash

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
            # dimensions = len(self.minhash_signature(output))
            dimensions = output.shape[0]
        self.vector_manager = Vector_Manager(dimensions=dimensions, save_path=self.save_path)
        # if dimensions == 100352:
        #     self.model = ResNet50(weights='imagenet', include_top=False)
        # else:
        #     base_model = ResNet50(weights='imagenet', include_top=False)
        #     x = base_model.output
        #     x = GlobalAveragePooling2D()(x)
        #     reduced_output = Dense(dimensions, activation=None)(x)
        #     self.model = Model(inputs=base_model.input, outputs=reduced_output)

        self.counter = 0
        # self.id_name_map = {}
        # self.id_name_map_file = os.path.join(self.save_path, "id_name_map.pkl")
        # if os.path.exists(self.id_name_map_file):
        #     with open(self.id_name_map_file, "rb") as f:
        #         self.id_name_map = pickle.load(f)

        # PCA setup
        # self.pca = PCA(n_components=4096)
        # if pca_fit_folder:
        #     self.fit_pca_on_folder(pca_fit_folder)

    def save(self):
        self.vector_manager.save()
        # with open(self.id_name_map_file, "wb") as f:
        #     pickle.dump(self.id_name_map, f)

    def close(self):
        self.save()
        self.vector_manager.close()

    # method: to get the MinHash signature of a vector
    def minhash_signature(self, vector):
        mh = MinHash(num_perm=self.num_perm)
        for item in vector:
            mh.update(str(item).encode('utf8'))
        print(mh.digest())
        return np.array(mh.digest(), dtype=np.uint8)
    
    # method: to extract image features
    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(224, 224))  # Bildgröße für ResNet50
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array).flatten()
        # return self.minhash_signature(features)
        return features

    # def fit_pca_on_folder(self, folder_path):
    #     """Liest alle Bilder aus einem Ordner, extrahiert deren Features und passt PCA an."""
    #     feature_list = []
    #     for file_name in os.listdir(folder_path):
    #         file_path = os.path.join(folder_path, file_name)
    #         if os.path.isfile(file_path):
    #             try:
    #                 features = self.extract_features(file_path)
    #                 feature_list.append(features)
    #             except Exception as e:
    #                 print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    #     if feature_list:
    #         features_matrix = np.vstack(feature_list)  # Matrix von Features
    #         self.pca.fit(features_matrix)
    #         print("PCA angepasst.")

    # def transform_features(self, features):
    #     """Wendet PCA-Transformation auf die extrahierten Features an."""
    #     features = np.array(features).reshape(1, -1)  # Sicherstellen, dass es eine Matrix ist
    #     reduced_features = self.pca.transform(features)
    #     return reduced_features.flatten()
    
    # function: to add a new image 
    def add_image(self, image_path :str, id :int):
        # image_name = os.path.basename(image_path)
        if not self.vector_manager.exists_id(id):
            features = self.extract_features(image_path)
            # features = self.transform_features(features)
            self.vector_manager.add(id=id, vector=features)
            # self.id_name_map[id] = image_name
            # self.save()
            self.counter += 1
            return 0
        else:
            return 1

    
    # function: get similar images 
    def get_similars(self, image_path, threshold=0.3, k=1):
        image_name = os.path.basename(image_path)
        features = self.extract_features(image_path)

        # mh = MinHash(num_perm=self.num_perm)
        # for item in features:
        #     mh.update(str(item).encode('utf8'))
        # print(mh.digest())

        # features = self.transform_features(features)
        similars = []
        (ids, similarities) = self.vector_manager.search(query_vector=features, k=k)
        ic(ids)
        ic(similarities)
        for i, id in enumerate(ids): 
            sim = similarities[i]
            if sim > threshold:
                # sim_img_name = self.id_name_map[id]
                print(f"Warn: Image '{image_name}' is up to {(round(sim*100, 2))}% similar to '{id}'")
                similars.append((i, id, sim))

                # ic(self.test_jaccard_map)
                # sim_features = self.test_jaccard_map[sim_img_name]
                # sim_mh = MinHash(num_perm=self.num_perm)
                # for item in sim_features:
                #     sim_mh.update(str(item).encode('utf8'))
                # print(sim_mh.digest())
                # ic(sim_mh.jaccard(mh))
        return similars


    # delete functions
    def delete_by_ids(self, ids: list):
        self.vector_manager.delete(ids=ids)

    # def delete_by_name(self, name: str):
    #     id_ = self.vector_manager.get_id_by_name(name)
    #     self.vector_manager.delete(ids=[id_])

