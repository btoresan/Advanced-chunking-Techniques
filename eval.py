from chunking_evaluation.chunking.fixed_token_chunker import FixedTokenChunker
from chunking_evaluation.chunking.recursive_token_chunker import RecursiveTokenChunker
from chunking_evaluation.chunking.cluster_semantic_chunker import ClusterSemanticChunker
from chunking_evaluation.chunking.kamradt_modified_chunker import KamradtModifiedChunker
from chunking_evaluation.chunking.base_chunker import BaseChunker
from chunking_evaluation.evaluation_framework.general_evaluation import GeneralEvaluation

from chromadb.utils import embedding_functions

from chunking_evaluation.utils import openai_token_count
import pandas as pd
from IPython.display import display, clear_output
import http.client
import os

from sentence_transformers import SentenceTransformer

from chromadb import Documents, Embeddings
from tqdm import tqdm

from DualSemanticChunker import DualSemanticChunker
from LumberChunker import LumberChunker

class DSChunker(BaseChunker):
    def __init__(self, min, max, model, std_mul, distance_dig):
        self._chunk_size = min
        self._chunk_overlap = 0
        self._min = min
        self._max = max
        self.model = model
        self.std_mul = std_mul
        self.distance_dig = distance_dig

    def split_text(self, text):
        # Custom chunking logic
        return DualSemanticChunker(text, 
                                   min_token_size=self._min, 
                                   max_token_size=self._max, 
                                   TransformerModel=self.model, 
                                   std_multiplier=self.std_mul,
                                   plot_splits=False,
                                   show_statisctics=False)
        
class Lumber_Chunker(BaseChunker):
    def split_text(self, text):
        return LumberChunker(text, model_type="DeepSeek")


class FreeEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/stsb-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Ensure that input is a list of strings
        if isinstance(input, str):
            input = [input]
        # Get embeddings for the input texts
        embeddings = self.model.encode(input, convert_to_numpy=True)
        # Convert embeddings to list if needed
        if embeddings.ndim > 1:
            embeddings = embeddings.tolist()
        return embeddings

ef = FreeEmbeddingFunction()

chunkers = [
    FixedTokenChunker(chunk_size=100, chunk_overlap=0, encoding_name="cl100k_base"),
    RecursiveTokenChunker(chunk_size=100, chunk_overlap=0, length_function=openai_token_count),
    KamradtModifiedChunker(avg_chunk_size = 100, embedding_function = ef),
    ClusterSemanticChunker(embedding_function=ef, max_chunk_size=100, length_function=openai_token_count),
    Lumber_Chunker(),
    DSChunker(min=1, max=100, model="sentence-transformers/stsb-mpnet-base-v2", std_mul=3.0, distance_dig=10),
]

# Initialize evaluation
evaluation = GeneralEvaluation()

results = []

# Initialize an empty DataFrame
df = pd.DataFrame()

# Display the DataFrame
display_handle = display(df, display_id=True)
for chunker in tqdm(chunkers):
    result = evaluation.run(chunker, ef)
    del result['corpora_scores']  # Remove detailed scores for brevity
    chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else 0
    chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else 0
    result['chunker'] = chunker.__class__.__name__ + f"_{chunk_size}_{chunk_overlap}"
    results.append(result)

    # Update the DataFrame
    df = pd.DataFrame(results)
    clear_output(wait=True)
    print(df)

print(df)
df.to_csv('results.csv', index=False)