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
