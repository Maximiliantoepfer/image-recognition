import os
import signal
import atexit
from Image_Manager import Image_Manager
from flask import Flask, request, jsonify

app = Flask(__name__)

print("Starting the Image_Manager")
# Elastic: dimensions=4096, threshold=0.6
# Faiss: dimensions=100352, threshold=0.3
image_manager = Image_Manager()
print("Started the Image_Manager")

# Directory, for the uploaded images
UPLOAD_FOLDER = 'upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild gefunden"}), 400

    image = request.files['image']
    image_id = request.form.get('image_id')
    image_name = f"{str(image_id)}.png" # request.form.get('image_name')

    if not image_id:
        return jsonify({"error": "image_id is necessary"}), 400

    try:
        image_id = int(image_id)
    except Exception as e:
        return jsonify({"error": f"image_id has to be a number!\n{e}"}), 400

    # Bild speichern
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)
    json, code = {}, 500
    if os.path.isfile(image_path):
        print("Adding Image")
        result = image_manager.add_image(image_path=image_path, id=image_id)
        if result == 0:
            print("Image added")
            json = {
                "message": "Image uploaded correctly",
                "image_id": image_id,
                # "image_name": image_name,
                "saved_path": image_path
            }
            code = 200
        else:
            print(f"ID '{image_id}' already taken, Image NOT added")
            json = {
                "message": f"ID '{image_id}' is already taken, Image NOT added",
                "image_id": image_id,
                # "image_name": image_name,
                "saved_path": image_path
            }
            code = 201
    else:
        json = {
            "error": "Internal Server Error: Image NOT uploaded correctly",
            "image_id": image_id,
            # "image_name": image_name,
            "saved_path": image_path
        }   
        code = 500
        
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return jsonify(json), code

@app.route('/upload-bulk', methods=['POST'])
def upload_bulk_images():
    if 'images' not in request.files:
        return jsonify({"error": "Kein Bild gefunden"}), 400
    
    try:
        # request.form.get('image_name')
        image_ids = request.form.get('image_ids').split(',')
        image_ids = [int(image_id.strip()) for image_id in image_ids]
        image_names = [f"{str(image_id)}.png" for image_id in image_ids]
        images = request.files.getlist('images')
        print(f"image ids {image_ids}")
    except Exception as e:
        return jsonify({"error": f"image_id has to be a number! {e}"}), 400
    
    if not image_ids:
        return jsonify({"error": "image_id is necessary"}), 400
    if len(image_ids) != len(images):
        print(f"ids {image_ids}, len {len(image_ids)} / images len {len(images)}") 
        return jsonify({"error": "length of image_ids and images are not equal"}), 400
    
    # Bilder speichern
    image_paths_to_add = []
    image_ids_to_add = []
    for i, image in enumerate(images):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_names[i])
        image.save(image_path)
        if os.path.isfile(image_path):
            image_paths_to_add.append(image_path)
            image_ids_to_add.append(image_ids[i])
    
    print("Adding Images")
    json, code = {}, 500
    result = image_manager.add_bulk_images(image_paths=image_paths_to_add, ids=image_ids_to_add)
    if result:
        print("Image added")
        json = result
        code = 200
    else:
        json = {
            "error": "Internal Server Error: Images NOT uploaded correctly",
            "image_ids": image_ids,
            # "image_name": image_name,
            "saved_paths": image_paths_to_add
        }   
        code = 500
    for image_path in image_paths_to_add:   
        if os.path.exists(image_path):
            os.remove(image_path)
    return jsonify(json), code  


@app.route('/similar-images', methods=['POST'])
def check_for_similar_images():
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild gefunden"}), 400

    image = request.files['image']
    image_id = request.form.get('image_id')
    image_name = f"{str(image_id)}.png" # request.form.get('image_name')
    threshold = request.form.get('threshold')
    k = request.form.get('top_k')

    if not image_id:
        return jsonify({"error": "image_id is necessary"}), 400

    try:
        image_id = int(image_id)
    except Exception as e:
        return jsonify({"error": f"image_id has to be a number\n{e}"}), 400
    
    try:
        if threshold: threshold = float(threshold)
        if k: k = int(k)
    except Exception as e:
        return jsonify({"error": f"threshold and k have to be numbers\n{e}"}), 400

    # Bild speichern
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)
    json, code = {}, 500
    if os.path.isfile(image_path):
        sims = None
        if threshold and k:
            sims = image_manager.get_similars(
                image_path=image_path, 
                threshold=threshold,
                k=k
            )
        elif threshold:
            sims = image_manager.get_similars(
                image_path=image_path, 
                threshold=threshold
            )
        elif k: 
            sims = image_manager.get_similars(
                image_path=image_path, 
                k=k
            )
        else:
            sims = image_manager.get_similars(image_path=image_path)

        if sims:
            print("Similar image already exists")
            print(sims)
            response_data = [
                {
                    "index": int(i), "image_id": str(id_),
                    # "image_name": str(image_name), 
                    "similarity": str(similarity)
                } for i, id_, similarity in sims
            ]
            json = {
                "message": "Similar image already exists",
                "data": response_data
            }
            code = 200
        else:
            json = {
                "message": "No similar images.",
                "image_id": image_id,
                # "image_name": image_name,
                "saved_path": image_path
            }
            code = 200
    else:
        json = {
            "error": "Internal Server Error: Image could NOT be opened correctly",
            "image_id": image_id,
            # "image_name": image_name,
            "saved_path": image_path
        }   
        code = 500    

    if os.path.exists(image_path):
        os.remove(image_path)

    return jsonify(json), code

@app.route('/delete', methods=['DELETE'])
def delete_images():
    from icecream import ic
    import math
    image_ids = request.form.get('image_ids')
    if not image_ids: 
        image_ids = request.args.get('image_ids')
    if image_ids:
        image_ids = image_ids.split(',')
    else:
        return jsonify({"error": "parameter 'image_id' or body field 'image_ids' is necessary"}), 400
    ic(image_ids)
    try: 
        image_ids = [int(image_id.strip()) for image_id in image_ids]
    except Exception as e:
        print(e)
        jsonify({"error": f"parameter 'image_ids={image_ids}' must be a list of integers"}), 400
    ic(image_ids)
    try:
        deleted_images = image_manager.delete_by_ids(ids=image_ids)
        log = f"Deleted image_ids: {deleted_images}, already not existing: {[_id for _id in image_ids if _id not in deleted_images]}"  
        print(log)
        return jsonify({"message": log}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "could not delete ids, maybe ids are already deleted"}), 400
    

def close_resources(*args):
    print("Server is closing ...")
    image_manager.close()

signal.signal(signal.SIGINT, close_resources)   # Ctrl+C
signal.signal(signal.SIGTERM, close_resources)  # z.B. `kill PID`
atexit.register(close_resources)

if __name__ == "__main__":
    print("\n_____________________ START _____________________\n")
    app.run(
        host='0.0.0.0', 
        port=5001, 
        debug=False
    ) 
    close_resources() 

