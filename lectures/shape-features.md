---
title: Representing Shapes
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: \today
---

# Content

- Chain codes
- Elliptical Fourier Descriptors
- Point Distribution Models

# Shapes

Shapes compactly describe objects in images.

::: notes
motivation for shape representation
:::

## Representing Shapes

A shape in an image could be represented using the coordinates of edge pixels.

## Representing Shapes

Pixel coordinates encode the _shape_ **and** the _location_

- describes the shape in the image coordinate frame
- same shape in two locations appears to be different

## Representing Shapes

We are not interested in where the shape is - just the representation of the shape itself.

# Chain Codes

Rather than represent edge pixels in terms of image coordinates, represent each pixel as a **direction**.

## Chain Codes

In which direction must we move to stay on the edge?

- Shape is a _sequence of directions_.
- This is a **chain code**.

## Connectivity

- Connectivity is the notion of pixels being connected.
- A path must pass through connected pixels.
- In which directions can we travel to stay on the path?

---

![4 and 8 connectivity](assets/svg/connectivity-4-8.svg)

---

![We will use 8 connectivity](assets/svg/connectivity-8.svg)

## Chain Code Example

![Encode this image](assets/svg/chain-code-example-01.svg)

---

::: columns

::::: column

![Encoding assumptions](assets/svg/chain-code-example-01.svg)

:::::
::::: column

Assume:

- 8 connectivity
- scan anti-clockwise
- start at left-most column, then top-most row
- edge pixels are black

:::::

:::

---

![The edge boundary](assets/svg/chain-code-example-02.svg)

---

![Resulting code: 6 6 7 0 1 1 2 3 5 3 5](assets/svg/chain-code-example-02.svg)

## Chain Codes

$6 6 7 0 1 1 2 3 5 3 5$

::: notes
this is the code... we had to take care of the starting point
:::

## Chain Codes

For invariance to starting location:

- compute the chain code and rotate so the code represents the smallest m-digit **shape-number**.
- $6 6 7 0 1 1 2 3 5 3 5 \rightarrow 0 1 1 2 3 5 3 5 6 6 7$

::: notes
Chain code is sensitive to the starting point!
:::

## Chain Codes

Chain codes _are_ **translation** invariant.

- Adding a constant value to the x, y coordinates does not change the shape.

Chain codes are **not** scale or rotation invariant.

## Chain Code Derivatives

Chain codes specify a direction in absolute terms.

- Eg. 0 represents East, regardless of current direction.

## Chain Code Derivatives

This idea can be extended to use a relative encoding.

- Represent the next direction as the number of turns required to stay on the shape boundary.

- In this case, 0 corresponds to straightforward.

- This is a chain code _derivative_ or differential chain code.

## Chain Code Derivatives

To compute the chain code derivative:

- Compute the difference between chain code elements.
- Take the result _modulo_ $n$ (the connectivity).

## Chain Code Derivatives

Need to be careful with the starting element.

- Common assumption is begin straightforward.
- Chain code wraps around, so starting code is relative to the last.

## Chain Code Derivatives

- Chain Code: $6 6 7 0 1 1 2 3 5 3 5$
- Derivative: $1 0 1 1 1 0 1 1 2 6 2$

NB: pay attention to modulus of negative numbers.

::: notes
from our previous example.
:::

## Chain Code Derivatives

Chain code derivative provides _rotational_ invariance for rotations of **90 degrees**.

## Chain Code Advantages

- compact representation - only boundary is stored
- invariant to translation
- easy to compute shape related features, e.g. area, perimeter, centroid

## Chain Code Disadvantages

- No true rotational invariance and no scale invariance.
- Extremely sensitive to noise, sub-sampling loses definition.
- Cannot have sub-pixel accurate descriptions, only 4 or 8-connectivity.

## Chain Code Disadvantages

Chain codes describe a specific instance of a shape.

- What about a class of non-rigid shapes?
- What about boundaries that are not closed?
- What about locating shapes automatically in images?

# Elliptical Fourier Descriptors

A **parametric** representation of a shape.

::: notes

will solve _some_ of these problems.

:::

## Aside: Fourier Series

A Fourier series is an expansion of a **periodic** function $f(x)$ in terms of an infinite **sum** of sines and cosines.

## Aside: Fourier Series

We can approximate non-periodic functions on a specific _interval_.

- by pretending the non-periodic part _is_ periodic **outside** the interval.

## Aside: Fourier Series

The Fourier series of a periodic function $f(x)$ of period $T$ is:

$$
f(x) = \frac{a_0}{2}
    + \sum_{k=1}^{\infty}
    \left[ a_k \cos \frac{2 \pi k x}{T}
    + b_k \sin \frac{2 \pi k x}{T} \right]
$$

for some set of Fourier coefficients $a_k$ and $b_k$ defined by the integrals:

$$
a_k = \frac{2}{T} \int_{0}^{T} f(x) \cos \frac{2 \pi k x}{T} \mathrm{d}x,~
b_k = \frac{2}{T} \int_{0}^{T} f(x) \sin \frac{2 \pi k x}{T} \mathrm{d}x.
$$
