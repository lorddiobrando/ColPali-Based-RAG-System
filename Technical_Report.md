# Technical Report: ColPali-Based Multi-Modal RAG System

**Date:** April 2026  
**Subject:** Architecture, Design Choices, Benchmarks, and Key Observations

---

## 1. Executive Summary

Traditional Retrieval-Augmented Generation (RAG) pipelines over rely on Optical Character Recognition (OCR) and text parsing heuristics to extract data from documents. This leads to catastrophic information loss when dealing with tables, infographics, multi-column layouts, and nested figures. 

This project implements an **OCR-free, multimodal RAG architecture** utilizing the **ColPali** approach (specifically powered by the `vidore/colSmol-256M` Vision-Language Model). By bypassing text extraction entirely, the system embeds raw document pages as rich contextual "visual patches," retrieving exact page images via *Late Interaction* scoring, and feeding those images directly into a multimodal generative LLM for faithful answer formulation.

## 2. System Architecture

The architecture consists of an offline ingestion pipeline and a real-time retrieval-generation backend hosted via a decoupled Flask/Nginx Docker stack (configured for Railway deployment).

### 2.1 Ingestion & Indexing
1. **Document Processing:** PDFs are converted into raw image pages (e.g., using the ViDoRe benchmark splits).
2. **Visual Embedding:** The encoder (`colSmol-256M`) processes each page. Instead of collapsing the page into a single dense vector, the model outputs a **multi-vector representation**—one dense vector for every visual patch (typically 1000+ patches per page).
3. **Storage:**
   - The heavy multi-vector embeddings are stored on disk in a binary cache.
   - A single **mean-pooled dense vector** (representing the global context of the page) is derived and inserted into **Qdrant**, serving as a rapid first-stage index.

### 2.2 Query Execution Pipeline
1. **Query Encoding:** A user's text query is embedded by the same ColSmol model, outputting multi-vectors (one per text token).
2. **Stage-1 Retrieval (Fast ANN):** The query's mean-pooled representation is executed against Qdrant using standard Cosine Similarity to fetch the top-50 candidate document pages.
3. **Stage-2 Reranking (Late Interaction):** The system loads the cached multi-vectors for the top-50 candidate pages. It computes a **MaxSim (Maximum Similarity)** score between every token in the query and every visual patch in the document. The top-K highest-scoring pages form the final retrieval set.
4. **Answer Generation:** The retrieved page images are dynamically base64-encoded and passed alongside the user's text prompt to `google/gemma-4-26b-a4b-it` (via OpenRouter) to generate a grounded, natural-language response.

---

## 3. Core Design Choices & Trade-offs

1. **OCR-Free Ingestion vs. Hardware Requirements**
   - *Advantage:* Zero parsing rules to maintain. Perfect preservation of semantic layout.
   - *Trade-off:* Generating and storing 1000+ full-dimensional vectors per document page introduces significant memory and storage overhead compared to traditional chunked text indexing.

2. **The Two-Stage Retrieval Compromise**
   - Pure ColBERT-style MaxSim scoring across millions of documents is computationally unfeasible without specialized infrastructure (like Vespa). 
   - *Choice:* Using Qdrant for a standard mean-pooled ANN search as a "funnel" before running MaxSim matrix multiplications natively in PyTorch on the candidates. This strikes a pragmatic balance between indexing speed, storage compatibility, and retrieval precision.

3. **Separating Stateful Components**
   - *Choice:* By using local `hdf5`/`safetensors` caches for the raw Multi-Vectors rather than forcing them all into the vector DB, we allow Qdrant to operate efficiently with standard 1D dense vectors, greatly reducing DB memory pressure.

---

## 4. Benchmark Performance

An end-to-end evaluation suite was engineered to systematically test the pipeline against the rigorous **ViDoRe Benchmark**, comprising 10 distinct datasets covering domains like government reports, shifting project data, infographics, and complex tables. 

*Results below are based on a stratified evaluation sample (N=50) relying strictly on visual embedding without metadata hints.*

### 4.1 Retrieval Quality
The retrieval step heavily dictates RAG viability. When identifying the single correct page out of 8,443 stored documents:
*   **NDCG@5:** `0.4408`
*   **Recall@1:** `36.0%`
*   **Recall@5:** `50.0%`
*   **Mean Reciprocal Rank (MRR):** `0.4207`

*(The system successfully isolated the ground-truth document to the single top spot 36% of the time, and found it in the top 5 half the time. It performed exceptionally well on pure Infographic datasets (`Recall@5 = 100%`) while struggling most heavily with extremely dense textual domains like `docvqa` (`Recall@5 = 0%`).)*

### 4.2 Answer Generation & Factual Grounding
Scored via LLM-as-a-Judge (1-5 scale):
*   **Conciseness:** `4.74 / 5`
*   **Completeness:** `2.90 / 5`
*   **Correctness:** `1.90 / 5`
*   **Hallucination Rate:** `48.3%`

*(Despite excellent conciseness, the `Correctness` and `Hallucination` metrics highlight the inherent challenge mid-tier LLMs still face when tasked strictly with "reading" complex image pixels. Over 86% of answers contained attempted citations, but the factual grounding strictly against the visual context often failed when reasoning over tabular matrices without a text anchor).*

---

## 5. Key Observations

1. **The "Cold Start" Inference Delays:** 
   Because ColSmol is integrated tightly into the runtime backend pipeline, its large dual-weights (Base backbone + ColPali Projection) inflict a 4–8 second penalty the very first time a query is made following a server restart as tensors are loaded into VRAM.
2. **True Multimodal Reasoning is the Bottleneck:** 
   The evaluation proves that the *retrieval* mechanism itself (Late Interaction on Image Patches) operates effectively (50% Recall@5 over 8.4k dense PDFs is strong for purely visual retrieval). However, the actual text-generation bottleneck has shifted. The generative LLM simply misreads data directly off the retrieved images roughly half the time.
3. **Future Strategic Improvements:** 
   The most impactful future step would be a hybrid approach—employing the visual MaxSim approach for purely *retrieval*, but incorporating lightweight OCR parsing *after* the optimal page is found to supplement the LLM's prompt context with raw text strings alongside the Base64 image, acting as a numeric constraint against hallucinations.
