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
2. Using the first shape as a reference, transform each example to align with the reference.
3. Compute the mean of the aligned shapes.
4. Align the mean with the first shape.
5. Transform shapes to match the adjusted mean.
6. If not converged, go to step 3.

**Convergence** is a _small_ change in the mean.

---

To align shapes:

$$
\begin{aligned}
\mathbf{x}_{1} &= \{ x_{11}, x_{12}, ..., y_{11}, y_{12}, ...y_{1n} \}^{T} \\
\mathbf{x}_{2} &= \{ x_{21}, x_{22}, ..., y_{21}, y_{22}, ...y_{2n} \}^{T}
\end{aligned}
$$

Scale and rotation is defined as:

$$
M(x, \theta) =
\begin{pmatrix}
\mathbf{x}_{i} s \cos \theta & - \mathbf{y}_{i} s \sin \theta  \\
\mathbf{x}_{i} s \sin \theta & + \mathbf{y}_{i} s \cos \theta
\end{pmatrix}
$$

and translation by:

$$
\mathbf{t} =
\begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
$$

## Aligning Shapes

The parameters for scaling, rotation and translation are unknown.

- They need to be calculated from the data.

## Aligning Shapes

Define a metric that measures how well two shapes are aligned.

- Use sum of squared differences between the shapes.

$$
E = (x_1 - M(s, \theta) \mathbf{x}_2 - t) ~W (x_1 - M(s, \theta) \mathbf{x}_2 - t)^T
$$

where $W$ is a diagonal weighting matrix.

---

We can alternatively write the equation as:

$$
\begin{split}
E = \sum_{i=1}^n & w_i
\left[
    \begin{pmatrix} x_{1i} \\ y_{1i} \end{pmatrix} -
    \begin{pmatrix}
        x_{2i} s \cos \theta -  y_{2i} s \sin \theta \\
        x_{2i} s \sin \theta +  y_{2i} s \cos \theta
    \end{pmatrix} -
    \begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
\right] \\
& \left[
    \begin{pmatrix} x_{1i} \\ y_{1i} \end{pmatrix} -
    \begin{pmatrix}
        x_{2i} s \cos \theta -  y_{2i} s \sin \theta \\
        x_{2i} s \sin \theta +  y_{2i} s \cos \theta
    \end{pmatrix} -
    \begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
\right]
\end{split}
$$

---

Let $a_x = s \cos \theta~$ and $~a_y = s \sin \theta$ and substitute:

$$
\begin{split}
E =  \sum_{i=1}^n & w_i
\left[
    \begin{pmatrix} x_{1i} \\ y_{1i} \end{pmatrix} -
    \begin{pmatrix}
        x_{2i} a_x -  y_{2i} a_y \\
        x_{2i} a_y +  y_{2i} a_x
    \end{pmatrix} -
    \begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
\right] \\
&\left[
    \begin{pmatrix} x_{1i} \\ y_{1i} \end{pmatrix} -
    \begin{pmatrix}
        x_{2i} a_x -  y_{2i} a_y \\
        x_{2i} a_y +  y_{2i} a_x
    \end{pmatrix} -
    \begin{pmatrix} t_{x} \\ t_{y} \end{pmatrix}
\right]
\end{split}
$$

---

then multiply:

$$
E = \sum_{i=1}^n w_i
\left[
    ( x_{1i} -  a_x x_{2i} + a_y y_{2i} - t_{x} )^2 +
    ( y_{1i} -  a_y x_{2i} - a_x y_{2i} - t_{y} )^2
\right]
$$

This is the _cost function_ we must **minimise**.

---

$$
E = \sum_{i=1}^n w_i
\left[
    ( x_{1i} -  a_x x_{2i} + a_y y_{2i} - t_{x} )^2 +
    ( y_{1i} -  a_y x_{2i} - a_x y_{2i} - t_{y} )^2
\right]
$$

We have four unknown parameters: $a_x$, $a_y$, $t_X$ and $t_y$.

- Differentiate with respect to each of parameter.
- Equate to zero.
- Solve the resulting system of equations.

---
