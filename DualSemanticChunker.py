import spacy
import numpy as np
from transformers import pipeline
import datasets
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Obj: Get embeddings for a list of sentences given a pipeline (check if sentence is too big)
# In: sentences: List of strings, embedding_pipeline: Pipeline
def get_sentence_embeddings(sentences, embedding_pipeline, max_length=100):
    tokenizer = embedding_pipeline.tokenizer

    #make sure the sentences have less tokens than max length
    for i, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(sentence)
        token_size = len(tokens)

        if token_size > max_length:
            parts = [''.join(tokens[i:i+max_length]) for i in range(1, token_size, max_length)]
            sentences[i] = parts[0]
            for part in parts[1:]:
                i+=1
                sentences.insert(i, part)

    #create a dataset to process the sentences in parallel
    dataset = datasets.Dataset.from_dict({"text":sentences})

    features = embedding_pipeline(dataset["text"], batch_size=16)
    embeddings = []
    for feature in features:
        token_embeddings = feature[0]
        sentence_embedding = np.mean(token_embeddings, axis=0)
        embeddings.append(sentence_embedding)

    return embeddings 

# Obj: Calculate a dynamic threshold for semantic chunking using the similarity matrix.
#      The threshold is calculated as the mean minus a multiplier times the standard deviation.
# In: similarity_matrix: List of float, std_multiplier: float
def dynamic_threshold(similarity_matrix, std_multiplier=0.5, distance=10):
    # Filter the similarity scores to only consider the target ones
    n = similarity_matrix.shape[0]
    rows, cols = np.indices((n, n))
    mask = (cols > rows) & (cols <= rows + distance)
    target_scores = similarity_matrix[mask]
    
    # Compute the mean and standard deviation of the filtered scores
    mean = np.mean(target_scores)
    std = np.std(target_scores)
    
    return mean - std_multiplier * std
    
# Obj: Get the similarity scores between the embeddings of the sentences.
# In: sentence_embeddings: List of np.array
def get_similarity_matrix(sentence_embeddings):
    if torch.cuda.is_available():
        matrix = torch.tensor(sentence_embeddings).float().cuda() 
    else:
        matrix = torch.tensor(sentence_embeddings).float()
    result = torch.matmul(matrix, matrix.t())

    magnitudes = np.linalg.norm(sentence_embeddings, axis=1)
    result= result/ np.outer(magnitudes, magnitudes)

    # Transfer result back to CPU and print it
    return result.cpu().numpy()
    

# Obj: Split a text into semantic chunks based on the similarity between the embeddings of the sentences.
def DualSemanticChunker(text, 
                      std_multiplier=1.0, 
                      min_token_size=100, 
                      max_token_size=500, 
                      SpacyModel="en_core_web_sm",
                      TransformerModel="bert-base-uncased",
                      static_threshold=False,
                      threshold_distance = 10,
                      threshold=0.5,
                      show_statisctics=False,
                      plot_splits=False,
                      device=0
                      ):
    
    #Avoid warnings
    datasets.logging.set_verbosity_info()
    
    #Loads the LLM models 
    nlp = spacy.load(SpacyModel)
    embedding_pipeline = pipeline("feature-extraction", model=TransformerModel, device=device)
    
    #Divides the text into sentences
    nlp.max_length = 100000 #Adjust to process large texts
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    #Tokenizes the sentences and gets the embeddings
    sentence_tokens = [nlp(sent) for sent in sentences]
    sentence_token_counts = [len(tokens) for tokens in sentence_tokens]

    embeddings = get_sentence_embeddings(sentences.copy(), embedding_pipeline)
    embeddings = [np.array(embedding) for embedding in embeddings]
    embeddings = np.array(embeddings)
    
    #Calculates the similarity matrix and the threshold
    similarity_matrix = get_similarity_matrix(embeddings)
    if not(static_threshold):
        threshold = dynamic_threshold(similarity_matrix, std_multiplier, threshold_distance)
    
    #Statistics
    splits = 0
    splits_by_size = 0
    
    #Chunks the text
    chunks = []
    splitting_points = [] #for plotting
    current_chunk = [sentences[0]]
    chunk_size = 1
    current_chunk_token_count = sentence_token_counts[0] #for min_token_size and max_token_size
    
    
    for i in range(1, len(sentences)):
        #Calculates the similarity scores for the current sentence
        part_scores = similarity_matrix[i, i - chunk_size:i]

        #Conditions for splitting
        is_below_threshold = any(part_score < threshold for part_score in part_scores)
        is_chunk_large_enought = current_chunk_token_count > min_token_size
        is_chunk_too_large = current_chunk_token_count >= max_token_size
        
        #If conditions met append the current chunk and start a new one
        if (is_below_threshold and is_chunk_large_enought) or is_chunk_too_large:
            chunk_size = 1
            splits += 1
            splitting_points.append(i)
            if is_chunk_too_large:
                splits_by_size += 1
            
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_chunk_token_count = sentence_token_counts[i]
        #Else append the sentence to the current chunk
        else:
            chunk_size += 1
            current_chunk.append(sentences[i])
            current_chunk_token_count += sentence_token_counts[i]
    
    #Append the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    #--------------------------STATISTICS AND PLOTTING--------------------------#

    if show_statisctics or plot_splits:
        docs = [nlp(chunk) for chunk in chunks]
        chunk_token_counts = [len(doc) for doc in docs]
    
        if show_statisctics:
            print("Splitting statistics:")
            print("\t-Text size: ", str(len(text.split())))
            print("\t-Number of chunks: ", str(splits + 1))
            print("\t-Total splits: ", str(splits))
            print("\t-Max chunk token size: ", str(max(chunk_token_counts)))
            print("\t-Min chunk token size: ", str(min(chunk_token_counts)))
            print("\t-Number of splits by threshold: ", str(splits - splits_by_size))
            print("\t-Number of splits by max size: ", str(splits_by_size))
            if splits != 0: print("\t-Split by threshold ratio: ", str((splits - splits_by_size)/(splits)))
            print("\t-Last split size: ", str(len(chunks[-1].split())))
            print("\t Threshold: ", str(threshold))

        if plot_splits:
            #Create a mask for the lower triangular part (below diagonal)
            mask = np.tril(np.ones_like(similarity_matrix, dtype=bool), k=-1)

            #Create a custom colormap: dark blue to red
            colors = [(0, 0, 0.4), (0, 0, 1), (1, 0, 0)]  # Dark blue-black, blue, red
            n_bins = 100  # Discretizes the interpolation into 100 steps
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

            #Create the figure and the heatmap
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(similarity_matrix, mask=mask, cmap=cmap, annot=False, square=True, cbar_kws={"shrink": .8})

            #Move the x-axis label to the top
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()

            #Add vertical lines to indicate the splits (with thinner lines)
            for split in splitting_points:
                plt.axvline(x=split, color='white', linestyle='--', linewidth=1)  # Thinner lines (linewidth=1)

            plt.xlabel('Block #')  # X-axis label on top
            plt.ylabel('Block #')
            plt.title('Similarity Matrix with Splits')  # Title

            # Reduce number of ticks on x and y axes for clarity
            xticks = ax.get_xticks()  # Get current tick positions
            ax.set_xticks(xticks[::2])  # Reduce the number of ticks (e.g., keep every second tick)

            yticks = ax.get_yticks()  # Get current tick positions
            ax.set_yticks(yticks[::2])  # Reduce the number of ticks on y-axis as well

            #Ensure the final feature is labeled
            xtick_labels = ax.get_xticklabels()  # Get the current x-axis tick labels
            xtick_labels[-1] = similarity_matrix.shape[1]  # Label the final feature with its index number
            ax.set_xticklabels(xtick_labels)

            ytick_labels = ax.get_yticklabels()  # Get the current y-axis tick labels
            ytick_labels[-1] = similarity_matrix.shape[0]  # Label the final feature on the y-axis with its index
            ax.set_yticklabels(ytick_labels)

            plt.show()

    return chunks
