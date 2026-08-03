import faiss
import numpy as np

from torch.utils.data import DataLoader

from sentence_transformers import (
    losses,
    InputExample
)

from utils import (
    EmbeddingService,
    load_dataset,
    save_pickle
)

###############################################################
# CONFIGURATION
###############################################################

DATASET = "School_Workflow_Dataset_100.xlsx"

MODEL_OUTPUT = "models/workflow_embedding_model"

FAISS_INDEX = "models/workflow.index"

METADATA = "models/workflow_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 16

EPOCHS = 10

###############################################################
# LOAD DATASET
###############################################################

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = load_dataset(DATASET)

print(df.head())

###############################################################
# BUILD TRAINING EXAMPLES
###############################################################

print()
print("=" * 60)
print("Preparing training examples...")
print("=" * 60)

training_examples = []

for _, row in df.iterrows():

    training_examples.append(

        InputExample(

            texts=[

                row["Prompt"],

                row["WorkflowText"]

            ]

        )

    )

print("Training examples:", len(training_examples))

###############################################################
# LOAD MODEL
###############################################################

print()
print("=" * 60)
print("Loading Sentence Transformer...")
print("=" * 60)

embedding_service = EmbeddingService(MODEL_NAME)

model = embedding_service.model

###############################################################
# TRAIN MODEL
###############################################################

print()
print("=" * 60)
print("Training...")
print("=" * 60)

embedding_service = EmbeddingService(
    "sentence-transformers/all-MiniLM-L6-v2"
)

###############################################################
# CREATE EMBEDDINGS
###############################################################

print()
print("=" * 60)
print("Encoding workflows...")
print("=" * 60)

workflow_embeddings = embedding_service.encode_batch(

    df["WorkflowText"].tolist()

)

###############################################################
# BUILD FAISS INDEX
###############################################################

print()
print("=" * 60)
print("Building FAISS index...")
print("=" * 60)

dimension = workflow_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(

    np.array(workflow_embeddings)

)

faiss.write_index(

    index,

    FAISS_INDEX

)

###############################################################
# SAVE METADATA
###############################################################

metadata = []

for _, row in df.iterrows():

    metadata.append(

        {

            "prompt": row["Prompt"],

            "workflow_json": row["WorkflowJson"],

            "workflow_text": row["WorkflowText"]

        }

    )

save_pickle(

    metadata,

    METADATA

)

###############################################################
# TEST
###############################################################

print()
print("=" * 60)
print("Testing retrieval...")
print("=" * 60)

query = input("Prompt: ")

query_embedding = embedding_service.encode(query)

scores, indices = index.search(

    np.array([query_embedding]),

    5

)

print()

print("=" * 60)
print("Top Matches")
print("=" * 60)

for score, idx in zip(scores[0], indices[0]):

    workflow = metadata[idx]

    print()

    print("Similarity :", score)

    print()

    print("Prompt")

    print(workflow["prompt"])

    print()

    print(workflow["workflow_text"])

print()

print("=" * 60)
print("Training Complete")
print("=" * 60)