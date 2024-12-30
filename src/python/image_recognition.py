import os
from Image_Manager import Image_Manager
from flask import Flask, request, jsonify

app = Flask(__name__)
# Elastic: dimensions=4096, threshold=0.6
# Faiss: dimensions=100352, threshold=0.3
image_manager = Image_Manager(threshold=0.3)

# Directory, for the uploaded images
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
        return jsonify({"error": "image_id und image_name are necessary"}), 400

    try:
        image_id = int(image_id)
    except Exception as e:
        return jsonify({"error": f"image_id has to be a number!\n{e}"}), 400

    # Bild speichern
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)
    if os.path.isfile(image_path):
        print("Adding Image")
        result = image_manager.add_image(image_path=image_path, id=image_id)
        if result == 0:
            print("Image added")
            return jsonify({
                "message": "Image uploaded correctly.",
                "image_id": image_id,
                "image_name": image_name,
                "saved_path": image_path
            }), 200
        else:
            print(f"ID '{image_id}' already taken. Image '{image_name}' NOT added")
            return jsonify({
                "message": f"ID '{image_id}' already taken. Image '{image_name}' NOT added.",
                "image_id": image_id,
                "image_name": image_name,
                "saved_path": image_path
            }), 200
    return jsonify({
        "error": "Internal Server Error: Image NOT uploaded correctly.",
        "image_id": image_id,
        "image_name": image_name,
        "saved_path": image_path
    }), 500

@app.route('/similar-images', methods=['POST'])
def check_for_similar_images():
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild gefunden"}), 400

    image = request.files['image']
    image_id = request.form.get('image_id')
    image_name = request.form.get('image_name')

    if not image_id or not image_name:
        return jsonify({"error": "image_id und image_name are necessary"}), 400

    try:
        image_id = int(image_id)
    except Exception as e:
        return jsonify({"error": f"image_id has to be a number!\n{e}"}), 400

    # Bild speichern
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)
    if os.path.isfile(image_path):
        sims = image_manager.get_similars(
            image_path=image_path, 
            k=5
        )
        if sims:
            print("Not automaticly adding image, because there already exists a similar one")
            print(sims)
            response_data = [
                {"index": int(i), "image_id": str(id_), "image_name": str(image_name), "similarity": str(similarity)}
                for i, id_, image_name, similarity in sims
            ]
            return jsonify({
                "message": "There are already similar images",
                "data": response_data
            }), 200

    return jsonify({
        "message": "No similar images.",
        "image_id": image_id,
        "image_name": image_name,
        "saved_path": image_path
    }), 200

def close_resources(*args):
    print("Server is closing...")
    image_manager.close()



if __name__ == "__main__":
    print("\n_____________________ START _____________________\n")
    app.run(debug=False)
    image_manager.close()

