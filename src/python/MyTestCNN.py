import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers, Model
from icecream import ic 


def create_pairs(images, labels):
    """Erstellt positive und negative Bildpaare mit den zugehörigen Labels."""
    pairs, pair_labels = [], []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx1 in range(len(images)):
        current_image = images[idx1]
        label = labels[idx1]
        
        # Positives Paar (selbe Klasse)
        positive_idx = np.random.choice(class_indices[label])
        pairs.append([current_image, images[positive_idx]])
        pair_labels.append(1)
        
        # Negatives Paar (andere Klasse)
        negative_label = (label + np.random.randint(1, num_classes)) % num_classes
        negative_idx = np.random.choice(class_indices[negative_label])
        pairs.append([current_image, images[negative_idx]])
        pair_labels.append(0)
        
    return np.array(pairs), np.array(pair_labels)

def create_base_model(input_shape=(64, 64, 3)):
    """Erstellt ein Basismodell zur Extraktion von Features."""
    # inputs = tf.keras.Input(shape=input_shape)
    # x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(128, activation='relu')(x)
    # outputs = layers.Dense(64, activation='relu')(x)  # Embedding-Vector
    # return Model(inputs, outputs)
       
    # ResNet50 als Basismodell ohne die top Schicht (Fully Connected Layer)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)  # GlobalAveragePooling anstelle von Flatten
    x = layers.Dense(128, activation='relu')(x)  # Optional eine Dense-Schicht
    outputs = layers.Dense(64, activation='relu')(x)  # Embedding-Vector
    
    return Model(inputs=base_model.input, outputs=outputs)

def create_siamese_network(base_model):
    """Erstellt das Siamese Network."""
    input1 = tf.keras.Input(shape=(64, 64, 3))
    input2 = tf.keras.Input(shape=(64, 64, 3))
    
    # Feature-Extraktion
    features1 = base_model(input1)
    features2 = base_model(input2)

    # Berechnung der L1-Distanz
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([features1, features2])
    
    # distance = layers.Lambda(lambda tensors: tf.keras.losses.cosine_similarity(tensors[0], tensors[1]))([features1, features2])
    # distance = layers.Lambda(lambda tensors: tf.keras.losses.cosine_similarity(tensors[0], tensors[1]))([features1, features2])
    # distance = layers.Reshape((1,))(distance)  # Umwandeln in eine Form, die für Dense geeignet ist
    
    # Klassifikation
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    return Model(inputs=[input1, input2], outputs=outputs)

def load_data_from_directory(base_dir, img_size=(64, 64)):
    """
    Liest Bildpaare und zugehörige Labels aus der gegebenen Ordnerstruktur ein, 
    wobei die Bildnamen beliebig sein können.
    
    Args:
        base_dir (str): Pfad zum Hauptordner der Daten.
        img_size (tuple): Zielgröße der Bilder (Breite, Höhe).

    Returns:
        pairs (np.ndarray): Array von Bildpaaren, Form (n, 2, img_size[0], img_size[1], 3).
        labels (np.ndarray): Array der Labels (0 oder 1), Form (n,).
    """
    pairs = []
    labels = []
    
    # Gehe durch alle Unterordner
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            # Liste aller Bilddateien im Unterordner
            images = [os.path.join(subdir_path, fname) for fname in os.listdir(subdir_path) 
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Prüfen, ob mindestens 2 Bilder und die Label-Datei existieren
            label_path = os.path.join(subdir_path, 'label.txt')
            if len(images) >= 2 and os.path.exists(label_path):
                # Lade die ersten beiden Bilder
                image1 = img_to_array(load_img(images[0], target_size=img_size)) / 255.0
                image2 = img_to_array(load_img(images[1], target_size=img_size)) / 255.0
                
                # Lade das Label
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
                
                # Füge das Paar und das Label hinzu
                pairs.append([image1, image2])
                labels.append(label)
    
    return np.array(pairs), np.array(labels)


def create(train_data, test_data, save_to=""):
    # base_model = create_base_model()
    base_model = create_base_model()
    siamese_model = create_siamese_network(base_model)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_pairs, train_labels = load_data_from_directory(base_dir=train_data)
    test_pairs, test_labels = load_data_from_directory(base_dir=test_data)

    siamese_model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, 
                    validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
                    epochs=10, batch_size=32)
    if save_to:
        model_json = siamese_model.to_json()
        filepath_architecture = f"{save_to}/model_architecture.json"
        filepath_weights = f"{save_to}/model_weights.h5"
        with open(filepath_architecture, "w") as json_file:
            json_file.write(model_json)
        siamese_model.save_weights(filepath_weights)
    return siamese_model

def load(from_path):
    with open(f"{from_path}/model_architecture.json", "r") as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights(f"{from_path}/model_weights.h5")
    return loaded_model



def predict_similarity_of_images(model, image1, image2, img_size=(64, 64)):
    # Lade die ersten beiden Bilder
    name_img1 = image1
    name_img2 = image2
    image1 = img_to_array(load_img(image1, target_size=img_size)) / 255.0
    image2 = img_to_array(load_img(image2, target_size=img_size)) / 255.0

    # Füge eine Batch-Dimension hinzu
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    
    # Vorhersage mit dem Modell
    prediction = model.predict([image1, image2])

    # Überprüfe die Form der Vorhersage
    print(f"Vorhersage Form: {prediction.shape}")

    # Zugriff auf den ersten Vorhersagewert
    if prediction.shape[0] == 1:
        print(f"Die Bilder {name_img1} und {name_img2} sind {'ähnlich' if prediction[0][0] > 0.5 else 'nicht ähnlich'}. Vorhersage: {prediction[0][0]:.2f}")
    else:
        print("Fehler: Die Vorhersage hat nicht die erwartete Form.")


def predict_similarity(model, img_dir, img_size=(64, 64)):
    """
    Liest zwei Bilder aus einem Ordner, kombiniert sie und gibt die Vorhersage aus,
    ob die Bilder ähnlich sind (0 oder 1).
    
    Args:
        model (tensorflow.keras.Model): Das trainierte Modell zur Vorhersage.
        img_dir (str): Pfad zum Ordner mit den Bildpaaren.
        img_size (tuple): Zielgröße der Bilder (Breite, Höhe).
    
    Returns:
        None
    """
    # Liste der Bilddateien im Verzeichnis
    images = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) 
              if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Prüfen, ob es mindestens zwei Bilder gibt
    if len(images) >= 2:
        for image1 in images:
            for image2 in images:
                predict_similarity_of_images(model=model, image1=image1, image2=image2, img_size=img_size) 
    else:
        print(f"Nicht genügend Bilder im Verzeichnis {img_dir} gefunden. Mindestens 2 Bilder erforderlich.")


# similarity = siamese_model.predict([test_image1, test_image2])
# print("Ähnlichkeits-Wahrscheinlichkeit:", similarity[0][0])
if "__main__" == __name__:
    model = create(train_data="src/python/train_img", test_data="src/python/test_img")
    print()
    print("______________________________________\n")
    print("-- Model created now showing output --")
    print("______________________________________\n")
    predict_similarity(model=model, img_dir="src/python/img")
    
