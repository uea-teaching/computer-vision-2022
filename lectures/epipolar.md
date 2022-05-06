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

- used to describe _geometric_ relations in image pairs
- efficient search and prediction of corresponding points
- search space reduces from 2D to 1D

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

## Epipoles and Epipolar Lines {data-auto-animate="true}

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

## Epipolar Geometry {data-auto-animate="true}

![Epipolar Geometry](assets/svg/epipolar.svg){width=80%}

::: notes
to review the figure, I'll just point out all these elements again...
:::
