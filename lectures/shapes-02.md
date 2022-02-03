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

## Landmarks

![shape landmarks](assets/svg/lips_pdm.svg)

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

- Differentiate with respect to each parameter.
- Equate to zero.
- Solve the resulting system of equations.

---

differentiate with respect to $t_x$:

$$
\frac{\delta E}{\delta t_x} = \sum_{i=1}^n w_i
(2(x_{1i} -  a_x x_{2i} + a_y y_{2i} - t_{x})(-1))
$$

---

equate to zero:

$$
\begin{aligned}
0 &= \sum_{i=1}^n w_i (- x_{1i} +  a_x x_{2i} - a_y y_{2i} + t_{x})
\end{aligned}
$$

---

distribute the weighting:

$$
\begin{aligned}
0 &= \sum_{i=1}^n w_i (- x_{1i} +  a_x x_{2i} - a_y y_{2i} + t_{x}) \\
0 &= - \sum_{i=1}^n w_i x_{1i} + a_x \sum_{i=1}^n w_i x_{2i} - a_y \sum_{i=1}^n w_i y_{2i} + t_{x} \sum_{i=1}^n w_i\\
\therefore \sum_{i=1}^n w_i x_{1i}  &= a_x \sum_{i=1}^n w_i x_{2i} - a_y \sum_{i=1}^n w_i y_{2i} + t_{x} \sum_{i=1}^n w_i
\end{aligned}
$$

---

let:

::: columns
::::: column
$$\sum_{i=1}^n w_i x_{1i} = X_1$$
$$\sum_{i=1}^n w_i x_{2i} = X_2$$
$$\sum_{i=1}^n w_i  = W$$
:::::

::::: column
$$\sum_{i=1}^n w_i y_{1i} = Y_1$$
$$\sum_{i=1}^n w_i y_{2i} = Y_2$$
:::::
:::

---

the expression for $\frac{\delta E}{\delta t_x}$ simplifies to:

$$X_1 = a_x X_2 - a_y Y_2 + t_x W$$

---

If we calculate the remaining derivatives, we can develop further substitutions:

$$
\begin{aligned}
C_1 &= \sum_{i=1}^n w_i (x_{1i}x_{2i} + y_{1i} y_{2i}) \\
C_2 &= \sum_{i=1}^n w_i (y_{1i}x_{2i} + x_{1i} y_{2i}) \\
Z   &= \sum_{i=1}^n w_i (x_{2i}^{2} + y_{2i}^{2})
\end{aligned}
$$

---

Finally, we have a system of linear equations:

$$
\begin{aligned}
X_1 &= a_x X_2 - a_y Y_2 + t_x W \\
Y_1 &= a_x Y_2 + a_y X_2 + t_y W \\
C_1 &= a_x Z   + t_x X_2 + t_y Y_2 \\
C_2 &= a_y Z   - t_x Y_2 + t_y X_2
\end{aligned}
$$

Solve for: $a_x$, $a_y$, $t_x$ and $t_y$.

## Aligning Shapes

This was a simplified version of Procrustes analysis.

- We did not constrain $M$ to be a rotation matrix.

Matlab has a procrustes function.

- We will compare the two methods in the lab.
