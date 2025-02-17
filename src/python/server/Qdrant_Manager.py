from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.preprocessing import normalize
import numpy as np

class Vector_Manager:
    def __init__(self, dimensions, collection_name="image_collection", host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Überprüfen, ob die Sammlung existiert, andernfalls erstellen
        exists = self.client.collection_exists(collection_name=self.collection_name)
        print(f"exists: {exists}")
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
            )
            # Sparse-Vektoren hinzufügen
            self.client.create_sparse_vector(
                collection_name=self.collection_name,
                vector_name="sparse_vector"
            )

    def add(self, id, vector):
        vector = normalize(np.array([vector]), norm='l2')[0]  # Normalisieren des Vektors
        point = PointStruct(id=id, vector=vector.tolist())
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query_vector, k=1):
        query_vector = normalize(np.array([query_vector]), norm='l2')[0]  # Normalisieren des Abfragevektors
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k
        )
        ids = [result.id for result in results]
        scores = [result.score for result in results]
        return ids, scores

    def delete(self, ids):
        self.client.delete(collection_name=self.collection_name, points_selector=ids)
