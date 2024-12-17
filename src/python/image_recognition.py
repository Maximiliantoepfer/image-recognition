import os
from Image_Manager import Image_Manager
from flask import Flask, request, jsonify

app = Flask(__name__)

# Ordner, in dem die hochgeladenen Bilder gespeichert werden
UPLOAD_FOLDER = 'src/python/upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild gefunden"}), 400

    image = request.files['image']
    image_id = request.form.get('image_id')
    image_name = request.form.get('image_name')

    if not image_id or not image_name:
        return jsonify({"error": "image_id und bild_name sind erforderlich"}), 400

    # Bild speichern
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    return jsonify({
        "message": "Bild erfolgreich hochgeladen",
        "image_id": image_id,
        "image_name": image_name,
        "saved_path": image_path
    }), 200


if __name__ == "__main__":
    # app.run(debug=True)

    image_manager = Image_Manager()

    # directory containing new images
    image_directory = "src/python/img"
    for image_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_file)
        if os.path.isfile(image_path):
            features_dict = image_manager.add_image(
                image_path=image_path, threshold=0.3
            )
    
    image_manager.close()

