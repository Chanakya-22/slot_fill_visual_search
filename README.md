# Multi-Modal Intent Classification and Slot Filling for Visual Search

This repository contains an initial PyTorch and Hugging Face multimodal project scaffold for visual search understanding. The project is based on the ideas captured in the zeroth review document provided in the workspace.

## Project Summary

The goal is to move beyond text-only intent classification and slot filling by incorporating visual context from the image paired with a text query. In visual search settings, important clues are often embedded in the uploaded image, making text-only processing insufficient.

## Problem Definition

The project targets a multi-modal input of the form:

- Text query: $Q_{text}$
- Image input: $I_{image}$

The expected outputs are:

- Intent prediction: $y_{intent}$
- Slot predictions: $Y_{slots}$

## Review-Inspired Approach

The proposed approach is centered on a context-aware co-attention fusion strategy:

- Text encoder: BERT-style language modeling
- Vision encoder: CLIP-style visual embeddings
- Fusion: cross-attention between text and image representations
- Outputs: intent classification and slot filling heads

## Repository Structure

The project currently includes:

- a mock data generation pipeline
- a multimodal dataset loader
- starter model modules under the src package
- training and prediction entry points

## Next Steps

The next phase will focus on:

- implementing the training loop
- refining the multimodal architecture
- evaluating the mock data pipeline
- expanding the dataset and model components
