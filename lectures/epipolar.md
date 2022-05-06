---
title: Epipolar Geometry
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: Spring 2022
---

# Contents {data-transition="convex"}

- Motivation
- Epipolar Geometry
- Epipolar Elements
- Point Correspondences

# Motivation {data-auto-animate="true}

Given $x'$ in the first image, **find** the corresponding point $x''$ in the second image.

::: notes
We can make a motivating statement.
For a while now we have been talking about finding features in images.
And lately, we have been talking about finding correspondences between
image features...
now, if we find a promising feature in the first image -
how do we go about finding the correspondence in the second image?
:::

## {data-auto-animate="true}

- Coplanarity constraint
- Intersection of two corresponding rays
- The rays lie in a 3D plane

![Coplanarity](assets/svg/coplanarity-1.svg){width=60%}

::: notes
A typical arrangement - one camera with projection centre on the
left and the other on the right.
When they observe the same point, the two rays will intersect in a plane.
This is the coplanarity constraint we used to derive the
fundamental matrix and the essential matrix.
This will also play a role in the epipolar geometry we will talk about now.

So when we can see the point in image 1, then we need to search for it in image 2?
Where do I search? The whole image? Or, can I reduce the search space?
:::

# Epipolar Geometry {data-auto-animate="true}

- describe _geometric_ relations in image pairs
- _efficient_ search and prediction of corresponding points
- search space reduced from 2D to 1D

::: notes
epipolar geometry can help us with the search.
This takes into account the baseline vectors, where points are mapped to,
how the epipolar plane intersects the image planes.
It turns out that this geometry can reduce the search from the whole image
to just one line in image 2.
This has a couple of advantages: most obviously, it is faster, but also it reduces
the possibility of making incorrect associations...
many images have multiple similar features.
:::

## Epipolar Geometry {data-auto-animate="true}

![Epipolar Geometry](assets/svg/epipolar.svg){width=80%}

::: notes
Lets look at some elements of this figure.
epipolar axis between projection centres
epipolar plane formed by the projection centres and the observed point
epipoles - the projection of the other camera's projection centre
epipolar lines - intersection of the epipolar plane with the image planes

Look at the projection of X in camera 1. We don't know how far away it is, it could be at U. But that point can only exist on the epipolar line in image 2.
:::

## Epipolar Geometry {data-auto-animate="true}

Epipolar elements:

::: incremental

- **epipolar axis** $\mathcal{B} = (O' O'')$
- **epipolar plane** $\mathcal{E} = (O' O'' X)$
- **epipoles** $e' = (O'')', e'' = (O')''$
- **epipolar lines** $\mathcal{L}'(X) = (O'' X)', \mathcal{L}''(X) = (O' X)''$

:::

::: notes
Again, with labels...
epipolar axis between projection centres
epipolar plane formed by the projection centres and the observed point, different for each observation.
epipoles - the projection of the other camera's projection centre
epipolar lines - projection of the ray from the other camera's projection centre to the observed point.
:::

## {data-auto-animate="true}

We can also write the **epipoles** as:

$$
e' = (O' O'') \cap \mathcal{E}',
\quad e'' = (O' O'') \cap \mathcal{E}''
$$

And the **epipolar lines** as:

$$
\mathcal{L}'(X) = \mathcal{E} \cap \mathcal{E}',
\quad \mathcal{L}''(X) = \mathcal{E} \cap \mathcal{E}''
$$

::: notes
this notation describes the intersections
:::

## {data-auto-animate="true}

![Epipolar Geometry](assets/svg/epipolar.svg){width=80%}

::: notes
to review the figure, I'll just point out all these elements again...
:::

## In the Epipolar Plane {data-auto-animate="true}

Assuming a distortion free lens:

::: incremental

- The projection centres $O'$ and $O''$.
- The observed point $X$.
- The epipolar lines, $\mathcal{L}'(X)$ and $\mathcal{L}''(X)$.
- The epipoles, $e'$ and $e''$.
- The image points $x'$ and $x''$.

:::

::: notes
all of these elements are in the epipolar plane.
:::

## In the Epipolar Plane {data-auto-animate="true}

- The projection centres $O'$ and $O''$.
- The observed point $X$.
- The epipolar lines, $\mathcal{L}'(X)$ and $\mathcal{L}''(X)$.
- The epipoles, $e'$ and $e''$.
- The image points $x'$ and $x''$.

**All lie in the epipolar plane** $\mathcal{E}$.

::: notes
This is especially important for the task of predicting correspondences.
And this is what allows us to restrict the search space to a line in image 2.
So, the epipolar plane is a constraining element for the search.
:::

## Predicting Point Correspondence {data-auto-animate="true}

Task: Predict the location of $x''$ given $x'$.

- For the epipolar plane $\mathcal{E} = (O'O''X)$
- The intersection of $\mathcal{E}$ and the second image plane $\mathcal{E}''$ yields the epipolar line $\mathcal{L}''(X)$
- The corresponding point $x''$ lies on that epipolar line $\mathcal{L}''(X)$.
- Search space is reduced from 2D to 1D.

::: notes
with all these elements all we need to do is search along the epipolar line.
In practice, it would be a good idea to lok either side by a pixel or two...
:::
