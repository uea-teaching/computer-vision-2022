---
title: Camera Calibration
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: Spring 2022
---

# Contents {data-transition="convex"}

- Zhang's Method
- Non-linear Distortion
- Three Point Algorithm

# Zhang's Method

A method of finding the **intrinsic** parameters of a camera.

- Zhang, Z., 2000. A flexible new technique for camera calibration. IEEE Transactions on pattern analysis and machine intelligence, 22(11), pp.1330-1334.

::: notes
the intrinsics are the parameters that belong to the camera - they remain the same
as we move the camera around.
:::

## Point mapping {data-auto-animate="true"}

![point mapping](assets/svg/parameters1.svg)

::: notes
in the previous lecture we worked on the projection of points onto the image plane,
and calculated the **projection matrix**.
:::

## Point mapping {data-auto-animate="true"}

![Point to pixel](assets/svg/pinhole-camera-world.svg)

::: notes
we can illustrate this graphically - and please note - I have positioned the
image plane in the imaginary position in front of the camera.
This is very common in the literature, and is valid due to the similar triangles
of the pinhole camera model.
:::

## Direct Linear Transformation {data-auto-animate="true"}

Compute the 11 _intrinsic_ **and** _extrinsic_ parameters of a camera.

::: {style="font-size:1.5em"}

$$
\textbf{x} = K R [ I_3 | - \textbf{X}_o ] \textbf{X}
$$

:::

::: notes
and here we have the five intrinsic parameters on K
and 3 rotation values in R, and the 3 translation -Xo
:::

## Zhang's Method {data-auto-animate="true"}

Compute the 5 _intrinsic_ parameters in $K$.

::: {style="font-size:1.5em"}

$$
\textbf{x} = K R [ I_3 | - \textbf{X}_o ] \textbf{X}
$$

:::

::: notes
Why do this? surely we are better computing all the parameters?
answer - DLT requires knowledge of the scene we may ot have.
:::

## Zhang's Method {data-auto-animate="true"}

Camera calibration using images of a **checkerboard**.

![calibration target](assets/png/chk_orig_1.png){width="60%"}

::: notes
this time we don't need to know 3D points in the scene - we use a checkerboard of known dimensions.
:::

## Checkerboard {data-auto-animate="true"}

- Board is of **known** size and structure.
- The board must be **flat**.

![Calibration targets](assets/png/chk_orig.png)

::: notes
we know how big the squares are, and we know how many squares there are.
it must be flat - this is an important property that we will exploit.
:::

## Checkerboard Method {data-auto-animate="true"}

Set the **world** coordinate system to the **corner** of the checkerboard.

::: incremental

- do this for _each_ image captured.
- all points lie on x/y plane with z=0

:::

![Detected corners](assets/png/chk_crn.png)

::: notes
using a corner detector, we find points on the checkerboard.
we use a 'trick' setting the xy plane to be parallel to the checkerboard,
with z pointing outward.
This means all the z coordinates are zero. We know how big the squares are in our printed pattern.
Knowing that z=0 changes the maths - let's see how!
:::

## Simplification {data-auto-animate="true"}

The $Z$ coordinate of each point is **zero**.

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}
    r_{11} & r_{12} & r_{13} & t_1 \\
    r_{21} & r_{22} & r_{23} & t_2 \\
    r_{31} & r_{32} & r_{33} & t_3
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ \color{red}{Z} \\ 1 \end{bmatrix}
$$

::: notes
This is our DLT equation.
The left matrix is the intrinsic matrix, and the right matrix is the extrinsic matrix.
so the Z (in red) is zero all the time.
:::

## Simplification {data-auto-animate="true"}

The last column of the rotation matrix has no effect on the system.

- we can delete these components from the system

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}
    r_{11} & r_{12} & \color{red}{r_{13}} & t_1 \\
    r_{21} & r_{22} & \color{red}{r_{23}} & t_2 \\
    r_{31} & r_{32} & \color{red}{r_{33}} & t_3
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ \color{red}{Z} \\ 1 \end{bmatrix}
$$

::: notes
because Z is always zero, the last column of the rotation matrix is not used.
it's something we don't need to estimate.
:::

## Simplification {data-auto-animate="true"}

- The $Z$ coordinate of each point is **zero**.
- Deleting the third column of $R$ gives us:

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}
    r_{11} & r_{12} & t_1 \\
    r_{21} & r_{22} & t_2 \\
    r_{31} & r_{32} & t_3
\end{bmatrix}
\begin{bmatrix} X \\ Y  \\ 1 \end{bmatrix}
$$

::: notes
every observed point leads to one of these equations.
:::

## Simplification {data-auto-animate="true"}

- Each observed point gives this equation.
- The _intrinsics_ persist for **all** images.
- The _extrinsics_ persist for **each** image.

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}
    r_{11} & r_{12} & t_1 \\
    r_{21} & r_{22} & t_2 \\
    r_{31} & r_{32} & t_3
\end{bmatrix}
\begin{bmatrix} X \\ Y  \\ 1 \end{bmatrix}
$$

::: notes
it is important to note that the intrinsics are the same for all images,
and the extrinsics are the same for one image, but all points.
:::

## Setting up the equations {data-auto-animate="true"}

Define a matrix $H$:

$$
H  = \begin{bmatrix} \textbf{h}_1 , \textbf{h}_2 , \textbf{h}_3 \end{bmatrix} =
\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}
    r_{11} & r_{12} & t_1 \\
    r_{21} & r_{22} & t_2 \\
    r_{31} & r_{32} & t_3
\end{bmatrix}
$$

One point generates this equation:

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = H \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
$$

::: notes
we can define a matrix H, the product of each 3x3 matrix, and we can also think of this as 3 column vectors h1, h2, h3.
:::

## Setting up the equations {data-auto-animate="true"}

For multiple point observations:

$$
\begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} =
\underset{3 \times 3}{H} \begin{bmatrix} X_i \\ Y_i \\ 1 \end{bmatrix},
\quad i = 1 ..., n
$$

Analogous to the _DLT_.

::: notes
and a reminder, the points are known, H is unknown.
now we do the same as we did with the DLT...
:::

## Parameter Estimation {data-auto-animate="true"}

We estimate a $3 \times 3$ homography instead of $3 \times 4$ projection.

$$
a_{x_i}^T \textbf{h} = 0, \quad a_{y_i}^T \textbf{h} = 0
$$

with:

$$
\begin{aligned}
\textbf{h}           &= vec(H^T) \\
a_{x_i}^T &= (-X_i, -Y_i, -1, 0, 0, 0, x_i X_i, x_i Y_i, x_i) \\
a_{y_i}^T &= (0, 0, 0, -X_i, -Y_i, -1, y_i X_i, y_i Y_i, y_i)
\end{aligned}
$$

::: notes
now very similar to the DLT - but we have 9 coefficients instead of 12.
we have these coefficient vectors ax ay, and the unknown matrix H.
we stack h to one long column vector...
:::

## Parameter Estimation {data-auto-animate="true"}

Solving the system of linear equations leads to an estimate of the parameters of $H$.

- We need to identify **at least** 4 points.
- $H$ has 8 Dof (degrees of freedom)
- each point provides 2 observations

We now have the parameters of $H$, how do we find $K$?

## Decompose Intrinsic Parameters {data-auto-animate="true"}

For the DLT, we could use QR decomposition to find the rotation matrix of the extrinsic parameters.

- We can not do this for Zhang's method.
- We eliminated part of $R$ earlier.

## Decompose Intrinsic Parameters {data-auto-animate="true"}

$$
H  = \begin{bmatrix} \textbf{h}_1 , \textbf{h}_2 , \textbf{h}_3 \end{bmatrix} =
\underbrace
{\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}}_{K}
\underbrace
{\begin{bmatrix}
    r_{11} & r_{12} & t_1 \\
    r_{21} & r_{22} & t_2 \\
    r_{31} & r_{32} & t_3
\end{bmatrix}}_{[\textbf{r}_1, \textbf{r}_2, \textbf{t}]}
$$

::: notes
as a reminder, we have removed a column of R, so we can't use QR decomposition.
We are not interested in the right half - but how do we get K?
:::

## Decompose Intrinsic Parameters {data-auto-animate="true"}

We need to extract $K$ from the matrix $H = K[\textbf{r}_1, \textbf{r}_2, \textbf{t}]$ we computed using SVD.

::: notes
so this is our task - there is no standard decomposition technique to do this.
:::

## Decompose Intrinsic Parameters {data-auto-animate="true"}

We need to extract $K$ from the matrix $H = K[\textbf{r}_1, \textbf{r}_2, \textbf{t}]$ we computed using SVD.

Four step process:

::: incremental

1. Exploit constraints of $K, \textbf{r}_1, \textbf{r}_2$
2. Define a matrix $B = K^{-T}K^{-1}$
3. Solve $B$ using another homogeneous linear system.
4. Decompose $B$.

:::

::: notes
first exploit properties we know - r1 r2 are columns of a rotation matrix - and have useful properties.

define B so the maths is consistent

we will then define a system of equations that will lead to B, based on the constraints we know from 1.

Once we have B, we need to decompose to get K.
:::

# Exploiting Constraints {data-auto-animate="true"}

What constraints do we have?

## Exploiting Constraints {data-auto-animate="true"}

$$
K = \begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}
$$

$K$ is **invertible**.

::: notes
k is upper triangular, all the elements on the diagonal are non zero - it is invertible.
we can take advantage of this...
:::

## Exploiting Constraints {data-auto-animate="true"}

$$
H  = \begin{bmatrix} \textbf{h}_1 , \textbf{h}_2 , \textbf{h}_3 \end{bmatrix} =
\underbrace
{\begin{bmatrix} c & s & x_H \\ 0 & c(1 + m) & y_H \\ 0 & 0 & 1 \end{bmatrix}}_{K}
\underbrace
{\begin{bmatrix}
    r_{11} & r_{12} & t_1 \\
    r_{21} & r_{22} & t_2 \\
    r_{31} & r_{32} & t_3
\end{bmatrix}}_{[\textbf{r}_1, \textbf{r}_2, \textbf{t}]}
$$

$$
[\textbf{r}_1, \textbf{r}_2, \textbf{t}] =
K^{-1} [\textbf{h}_1 , \textbf{h}_2 , \textbf{h}_3]
$$

$$
\Rightarrow \textbf{r}_1 = K^{-1} \textbf{h}_1, \quad \textbf{r}_2 = K^{-1} \textbf{h}_2
$$

::: notes
so, multiply both sides by K-inverse gives us identity on the right...
swap over giving r1, r2, t = k-inverse h1 h2 h3

reminder - h is known - we dont know K or r1, r2
but we do know r1 r2 are columns of a rotation matrix.
:::

## Exploiting Constraints {data-auto-animate="true"}

As $[\textbf{r}_1 , \textbf{r}_2 , \textbf{r}_3]$ are the columns of a rotation matrix, they form an orthonormal basis.

$$
\textbf{r}_1^T \textbf{r}_2 = 0, \quad ||\textbf{r}_1|| = ||\textbf{r}_2|| = 1
$$

::: notes
so the dot product of r1 and r2 is zero - and the length of r1 and r2 are equal and are unit.
:::

## Exploiting Constraints {data-auto-animate="true"}

$$
\textbf{r}_1 = K^{-1} \textbf{h}_1, \quad \textbf{r}_2 = K^{-1} \textbf{h}_2, \quad
\textbf{r}_1^T \textbf{r}_2 = 0, \quad ||\textbf{r}_1|| = ||\textbf{r}_2|| = 1
$$

$$
\textbf{h}_1^T K^{-T} K^{-1} \textbf{h}_2 = 0
$$

$$
\begin{aligned}
\textbf{h}_1^T K^{-T} K^{-1} \textbf{h}_1 = \textbf{h}_2^T K^{-T} K^{-1} \textbf{h}_2 \\[10pt]
\textbf{h}_1^T K^{-T} K^{-1} \textbf{h}_1 - \textbf{h}_2^T K^{-T} K^{-1} \textbf{h}_2 = 0
\end{aligned}
$$

::: notes
inverse transpose means the transpose of the inverse.
here we substitute all the terms for r1, r2 and set the equations to zero.
:::

## Exploiting Constraints {data-auto-animate="true"}

$$
\textbf{h}_1^T K^{-T} K^{-1} \textbf{h}_2 = 0
$$

$$
\textbf{h}_1^T K^{-T} K^{-1} \textbf{h}_1 - \textbf{h}_2^T K^{-T} K^{-1} \textbf{h}_2 = 0
$$

::: notes
so these are our two equations that relate our knowns in the h vectors, and our unknowns in K.
now, lets define a matrix B and do a simple variable substitution.
:::

## Exploiting Constraints {data-auto-animate="true"}

Define a matrix $B := K^{-T}K^{-1}$

$$
\textbf{h}_1^T B \textbf{h}_2 = 0
$$

$$
\textbf{h}_1^T B \textbf{h}_1 - \textbf{h}_2^T B \textbf{h}_2 = 0
$$

::: notes
so now we have all our unknowns in B.
:::

## Exploiting Constraints {data-auto-animate="true"}

From $B$ the calibration matrix can be recovered using _Cholesky_ decomposition.

$$
B = \begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{bmatrix}
$$

$$
chol(B) = AA^T \Rightarrow A = K^{-T}
$$

If we know $B$, we can recover the calibration matrix $K$.

::: notes
There is a known method for $B$ - Cholesky decomposition.
We can decompose a matrix times its transpose. So if we know $B$, we can recover $K$.
How can we find B?
:::

## Exploiting Constraints {data-auto-animate="true"}

What do we have so far?

$$
\textbf{h}_1^T B \textbf{h}_2 = 0
$$

$$
\textbf{h}_1^T B \textbf{h}_1 - \textbf{h}_2^T B \textbf{h}_2 = 0
$$

- Matrix $B$, which is symmetric positive, so 6 unknowns.
- $\textbf{h}$ are known.
- Two equations that relate $B$ and $\textbf{h}$.

::: notes
B is symmetric so has 6 unknowns not 9.
2 equations that relate B and h and sets them to zero.
These 2 equations are similar to the DLT equations - we can reform it to a coefficient vector, times an unknown vector. We have that here too, so we will do the same trick.
Let's set up the equations...
:::

## Exploiting Constraints {data-auto-animate="true"}

Define a vector $\textbf{b} = (b_{11}, b_{12}, b_{13}, b_{22}, b_{23}, b_{33})$

$$
B = \begin{bmatrix}
    \color{red}{b_{11}} & \color{red}{b_{12}} & \color{red}{b_{13}} \\
    b_{12} & \color{red}{b_{22}} & \color{red}{b_{23}} \\
    b_{13} & b_{23} & \color{red}{b_{33}}
    \end{bmatrix}
$$

There are 6 unknowns in $B$, because it is symmetric.

## Exploiting Constraints {data-auto-animate="true"}

Construct a system of equations $V\textbf{b}=0$ exploiting our constraints.

$$
v^{T}_{12}\textbf{b} = 0,  \quad v^{T}_{11}\textbf{b} - v^{T}_{22}\textbf{b} = 0
$$

::: notes
This is similar to earlier, with the coefficients in v constructed from the knowns in H.
:::

## Matrix $V$ {data-auto-animate="true"}

The matrix $V$ is given by:

$$
V = \begin{bmatrix} v^{T}_{12} \\ v^{T}_{11} - v^{T}_{22} \end{bmatrix}, \quad
with \quad v_{ij} =
\begin{bmatrix}
h_{1i}h_{1j} \\ h_{1i}h_{2j}+h_{2i}h_{1j} \\
h_{3i}h_{1j}+h_{1i}h_{3j} \\ h_{2i}h_{2j} \\
h_{3i}h_{2j}+h_{2i}h_{3j} \\ h_{3i}h_{3j}
\end{bmatrix}
$$

::: notes
so we use the elements of h to construct the matrix v.
:::

## Matrix $V$ {data-auto-animate="true"}

For **each** image we get:

$$
\begin{bmatrix} v^{T}_{12} \\ v^{T}_{11} - v^{T}_{22} \end{bmatrix} \textbf{b} = 0
$$

::: notes
v is a 2x6 matrix. b is a 6x1 vector.
For every image we get this equation.
recall: we needed 4 points in each image to get H.
If you are working with the original paper we are at the end of section 3.
:::

## Matrix $V$ {data-auto-animate="true"}

For multiple images we stack the matrices to a $2n \times 6$ matrix:

$$
\begin{bmatrix} v^{T}_{12} \\ v^{T}_{11} - v^{T}_{22} \\
\dots \\
v^{T}_{12} \\ v^{T}_{11} - v^{T}_{22}
\end{bmatrix} \textbf{b} = 0
$$

We need to solve the linear system of $V\textbf{b}=0$ to find $b$ and hence $K$.

::: notes
for the third time today, we are at the point when we need to solve this system, using SVD.
:::

## Solving the Linear System {data-auto-animate="true"}

The system $V\textbf{b}=0$ has a trivial solution when $\textbf{b}=0$ which will not provide a valid matrix $B$.

- Apply additional constraint $||\textbf{b}|| = 1$ .

## Solving the Linear System {data-auto-animate="true"}

Real world measurements are noisy.

- Find the solution that minimises the least squares error:

$$
b^* = arg\underset{b}{min}||V \textbf{b}|| \quad \text{with} \quad ||\textbf{b}|| = 1
$$

Use SVD and choose the singular vector corresponding to the smallest singular value.

::: notes
Once we have b, we can rebuild the matrix B, and use Cholesky decomposition to recover K directly.
:::

## Minimum Requirements {data-auto-animate="true"}

- At least 4 points in each target image.
- Each target image gives _two_ equations.
- $B$ has 6 DoF so we need 3 _different_ views of the target.
- Solve $V\textbf{b}=0$ using SVD to compute $K$.

::: notes
we take photos of our targets and we need at least 4 points in each image.
We know the size, and that the target is flat.
From 4 points we compute a matrix H for each image.
We construct a B matrix for all images.
SVD gives us the solution directly of Vb=0, then we use cholesky to get K.
:::

# Non-Linear Distortion {data-auto-animate="true"}

How to deal with non-linear distortion?
