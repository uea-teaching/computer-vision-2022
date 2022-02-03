---
title: Point Distribution Models
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: \today
---

# Content

- Point Distribution Models
- Procrustes Analysis
- Principal Component Analysis

# Point Distribution Models

A _generative_ statistical model of the variation of the shape of an object.

## Point Distribution Models

# Procrustes Analysis

Normalise the shapes for _pose_ using generalised Procrustes analysis.

## Aligning Shapes

1. Translate each example so it is centred on the mean.
2. Using the first shape as a reference, rotate and scale each example to align with the reference.
3. Compute the mean of the aligned shapes.
4. Align the mean with the first shape.
5. Rotate shapes to match the adjusted mean.
6. If not converged, go to step 3.

**Convergence** is a _small_ change in the mean.

To align shapes:

$$
\begin{aligned}
\mathbf{x}_{1} &= \{ x_{11}, x_{12}, ..., y_{11}, y_{12}, ...y_{1n} \}^{T} \\
\mathbf{x}_{2} &= \{ x_{21}, x_{22}, ..., y_{21}, y_{22}, ...y_{2n} \}^{T}
\end{aligned}
$$

Scale and rotation is defined as:

$$
M(x, \theta) \mathbf{x}_{2} =
\begin{pmatrix}
\mathbf{x}_{2i} s \cos \theta & - \mathbf{y}_{2i} s \sin \theta  \\
\mathbf{x}_{2i} s \sin \theta & + \mathbf{y}_{2i} s \cos \theta
\end{pmatrix}
$$

and translation by:

$$
\mathbf{t} =
\begin{pmatrix} x_{2i} \\ y_{2i} \end{pmatrix} +
\begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
$$

---
