from chunking_evaluation.chunking import FixedTokenChunker, RecursiveTokenChunker, ClusterSemanticChunker, LLMSemanticChunker, KamradtModifiedChunker
from chunking_evaluation import GeneralEvaluation, SyntheticEvaluation, BaseChunker
from chunking_evaluation.utils import openai_token_count
from chromadb.utils import embedding_functions
import pandas as pd
from IPython.display import display, clear_output
import http.client
import os

from sentence_transformers import SentenceTransformer

from chromadb import Documents, Embeddings
from tqdm import tqdm

from splitter2 import FixedWindowSplitter

class FixedWindowChunker(BaseChunker):
    def __init__(self, min, max):
        self._chunk_size = min
        self._chunk_overlap = 0
        self._min = min
        self._max = max

    def split_text(self, text):
        # Custom chunking logic
        return FixedWindowSplitter(text, min_token_size=self._min, max_token_size=self._max)


class FreeEmbeddingFunction:
    def __init__(self, model_name="all-mpnet-base-v2"):
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
    FixedTokenChunker(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    RecursiveTokenChunker(chunk_size=200, chunk_overlap=0, length_function=openai_token_count),
    KamradtModifiedChunker(avg_chunk_size = 200, embedding_function = ef),
    ClusterSemanticChunker(embedding_function=ef, max_chunk_size=200, length_function=openai_token_count),
    FixedWindowChunker(min=1, max=200),

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