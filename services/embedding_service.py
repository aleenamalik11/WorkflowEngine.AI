from sentence_transformers import SentenceTransformer


###############################################################
# Embedding Service
###############################################################

class EmbeddingService:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)


    def encode(self, text):

        if not text:
            text = ""

        return self.model.encode(
            text,
            normalize_embeddings=True
        )
