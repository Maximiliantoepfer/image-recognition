from elasticsearch import Elasticsearch, helpers
from configparser import ConfigParser
from icecream import ic

config = ConfigParser()
config.read('src/python/elastic.conf')
elastic_host = config.get('Elastic', 'HOST')
elastic_ca_cert = config.get('Elastic', 'CA_CERT')
api_key = config.get('Elastic', 'API_KEY')

es = Elasticsearch(
    elastic_host,
    ca_certs=elastic_ca_cert,
    api_key=api_key
)

# Indexieren eines Vektors
def index_image(image_name, image_vector):
    doc = {
        "name": image_name,
        "feature": image_vector
    }
    es.index(index="image_vectors", document=doc)

# Suche nach ähnlichen Bildern
def search_similar_images(query_vector, top_k=5):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'feature') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = es.search(
        index="image_vectors",
        query=script_query,
        size=top_k
    )
    return response["hits"]["hits"]

