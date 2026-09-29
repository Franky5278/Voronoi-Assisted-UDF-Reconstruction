# Voronoi-Assisted UDF Reconstruction

## Overview

This repository documents a reproduction study of a Voronoi-assisted unsigned distance field (UDF) reconstruction pipeline for unoriented point clouds.

The project focuses on understanding and reproducing the major geometric stages of the method, from Voronoi construction and bisector sampling to bi-directional normal optimization, diffusion and UDF integration.

## Reproduction Pipeline

Input Point Cloud
→ Voronoi Diagram Construction
→ Voronoi Bisector Extraction
→ Bisector-Surface Sampling
→ Bi-directional Normal Optimization
→ Normal Diffusion
→ UDF Integration
→ Surface Reconstruction

## Implemented Components

- Point-cloud preprocessing
- Voronoi diagram construction
- Voronoi bisector extraction
- Surface sampling on bounded bisectors
- Bi-directional normal initialization
- Normal optimization
- Normal-field validation
- Diffusion-based field construction
- UDF integration
- Reconstruction visualization

## Experiments

Experiments were conducted using:

- Stanford Bunny
- Synthetic geometric examples
- Different sampling densities

The reproduction study analyzes:

- Normal-field convergence
- Effect of sparse sampling
- Voronoi bisector geometry
- UDF reconstruction behavior
- Failure cases under ambiguous or complex geometry

## Example Pipeline

```text
Stanford Bunny Point Cloud
        ↓
Voronoi Diagram
        ↓
Bisector Polygon Sampling
        ↓
Bi-directional Normals
        ↓
Normal Optimization
        ↓
Diffusion
        ↓
UDF
        ↓
Surface Reconstruction
