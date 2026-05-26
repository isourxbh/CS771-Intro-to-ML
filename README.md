# CS771: Introduction to Machine Learning - Major Assignment

## Overview
This repository contains my solutions for the CS771 Introduction to Machine Learning major assignment. The project tackles two distinct problems: implementing a custom kernel for Semi-Parametric Regression and recovering delays by inverting a XOR Arbiter PUF linear model.

## Problem 1: Semi-Parametric Regression (Staff Hiring Predictor)
* **Objective:** Designed a custom kernel to convert a semi-parametric regression problem into a purely non-parametric kernel ridge regression problem.
* **Approach:** Built a custom Gram matrix by combining an outer product of the primary feature (video length) with a 3-degree polynomial kernel applied to the auxiliary features (popularity and difficulty). 
* **Tools:** `numpy`, `sklearn.kernel_ridge`, `sklearn.metrics.pairwise`.

## Problem 2: Delay Recovery in XOR Arbiter PUFs
* **Objective:** Decoded a 1089-dimensional linear model of a XOR arbiter PUF to recover the 256 non-negative delays.
* **Approach:** * The linear model is a Kronecker product of two individual PUF models.
    * I applied **Singular Value Decomposition (SVD)** to perform a rank-1 factorization, effectively isolating the individual models from the 1089-dimensional space.
    * Transformed the resulting components into valid, non-negative timing delays through algebraic substitution and zero-bounding.

## Files
* `submit.py`: Contains the core logic (`my_kernel` and `my_decode`).
* `Major Assignment 1-2 - CS771 2025-26-I.pdf`: Detailed problem statements and mathematical foundations.