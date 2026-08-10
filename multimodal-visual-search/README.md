# Multi-Modal Intent Classification and Slot Filling for Visual Search

This repository is being initialized for a PyTorch and Hugging Face multimodal project focused on visual search understanding. The initial structure and README are based on the project review document provided in the workspace.

## Project Summary

The goal of this project is to move beyond traditional text-only intent classification and slot filling by incorporating visual context from an image alongside a text query. In visual search settings, important clues are often embedded in the uploaded image, making text-only processing insufficient.

## Problem Definition

The project targets a multi-modal input of the form:

- Text query: $Q_{text}$
- Image input: $I_{image}$

The model is expected to produce:

- Intent prediction: $y_{intent}$
- Slot predictions: $Y_{slots}$

This makes the problem a combination of multi-class classification and sequence tagging.

## Review-Inspired Approach

The review document outlines a direction centered on a context-aware co-attention fusion strategy:

- Text encoder: BERT-style language modeling
- Vision encoder: CLIP-style visual embeddings
- Fusion: cross-attention between text and image representations
- Outputs: intent classification and slot filling heads

This design is intended to reduce modality conflict and allow the model to focus on the most relevant visual regions for each text token.

## Baselines and Comparison

The project will compare the proposed approach with:

1. A text-only BERT baseline
2. A simple early-fusion multimodal baseline using concatenation

The expected evaluation will focus on:

- Intent accuracy
- Macro F1 for intent classification
- Sequence F1 for slot filling

## Repository Structure

The scaffold currently contains:

- data/raw and data/processed for dataset storage
- src for model and data utilities
- baseline and cross-attention model modules under src/models
- starter scripts for mock data generation, training, and prediction

## Next Steps

The next phase will involve:

- implementing the dataset pipeline
- adding mock or curated multimodal samples
- building the baseline and cross-attention models
- training and evaluating the pipeline

## Notes

This README is an initial starter draft based on the zeroth review PDF and will evolve as the implementation progresses.
