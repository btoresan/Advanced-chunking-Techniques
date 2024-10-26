# Dual Semantic Chunker (DSC)
*An advanced, semantically-driven chunking tool for enhanced Information Retrieval (IR) performance.*

---

## Overview

**Dual Semantic Chunker (DSC)** is a novel approach for segmenting text into meaningful, coherent chunks using semantic representations. Unlike traditional chunking methods (fixed-length or syntactic-based), DSC splits text based on meaning and context, preserving the semantic integrity of each chunk. The resulting segments lead to improved retrieval quality, as measured by metrics like F1 and DCG@k, compared to traditional chunking approaches.

This repository contains the source code, configurations, and evaluation framework used to implement and assess DSC across various datasets.

## Features

- **Two-Step Chunking Process**:
  - Initial segmentation into blocks.
  - Semantic evaluation to select optimal splitting points.
- **Semantic Representation**: Utilizes pre-trained models to ensure each chunk represents a logical, complete unit.
- **High Retrieval Performance**: Demonstrated improvements in retrieval quality metrics (F1, DCG@k) on benchmark datasets.
- **Flexible Parameterization**: Adjustable parameters allow tuning for recall or precision emphasis.

## Methodology

The DSC model operates in two main stages:

1. **Initial Block Splitting**: Text is divided into initial blocks using spaCy's `en_core_web_sm` model.
2. **Semantic Split Selection**: Semantic embeddings of blocks are analyzed to determine optimal chunk boundaries. This involves setting a similarity threshold and selecting split points based on the semantic structure of the text.

## Repository Contents

- **`chunking_evaluation/`**: Modified chromadb evaluation to support ranking-wise metrics.
- **`results/`**: Results tables of our experiments
- **`eval.py/`**: Code using the evaluation to compare diferent chunking methods with open models.
- **`DualSemanticChunker/`**: Implementation of our approach to semantic chunking
- **`requirements.txt/`**: Library depencies organizes to facilitate setup
