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

## {data-auto-animate="true}

- Coplanarity constraint
- Intersection of two corresponding rays
- The rays lie in a 3D plane

![Coplanarity](assets/svg/coplanarity-1.svg){width=80%}

::: notes
We want to exploit what we have just learnt about coplanarity.
:::

# Epipolar Geometry {data-auto-animate="true}

- used to describe _geometric_ relations in image pairs
- efficient search and prediction of corresponding points
- search space reduces from 2D to 1D

## Epipolar Geometry {data-auto-animate="true}

![Epipolar Geometry](assets/svg/epipolar.svg){width=80%}
