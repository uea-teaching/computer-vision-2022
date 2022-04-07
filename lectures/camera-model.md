---
title: The Camera
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: Spring 2022
---

# Contents {data-transition="convex"}

- Camera Model
- Intrinsic and Extrinsic Parameters
- Direct Linear Transformation

::: notes
We have talked a lot about images, but not much on how we obtain images using a camera.
We will discuss the parameters of a pinhole camera model,
that serves well for computer vision tasks.
In the first part - we will then show the relationship between points in the world and points in the image.
:::

# The Camera

!["Sallie Gardner," owned by Leland Stanford; ridden by G. Domm, running at a 1:40 gait over the Palo Alto track, 19th June 1878.](assets/jpg/horse.jpg){width="80%"}

::: notes
who would disagree if I said the camera was one of the most important inventions in the history of science?
Here Edweard Muybridge shows definitively that a horses feet al leave the ground when galloping.
:::

## The Camera {data-auto-animate="true"}

Cameras measure light **intensities**.

- the sensor counts photons arriving at the pixel
- each pixel corresponds to a direction in world space

::: notes
What do cameras measure?
:::

## The Camera {data-auto-animate="true"}

Cameras can also be seen as _direction_ measurement devices.

- we are often interested in geometric properties of a scene
- an object reflects light to a specific location on the sensor
- Which 3D point is mapped to which pixel?

::: notes
light falls on an object, is reflected back to the camera, and then to a pixel.
which point in space maps to which position on the sensor?
this is information used to perform geometric measurements.
:::

## The Camera {data-auto-animate="true"}

How do we get the point observations?

- _keypoints_ and _features_
- SIFT, ORB, etc.
- **locally** distinct features

::: notes
We get these observations by detecting features in an image, finding corners, blobs, etc.
Thes are locally distinct features...
:::

## The Camera {data-auto-animate="true"}

Features identify points mapped from the 3D world to the 2D image.

::: notes
We assume this point has been mapped to the 2D image plane, and we want to reconstruct this point in the environment.

We often use a small number of such keypoints - maybe a few hundred per image - much less than the number of pixels in the image.

So the camera records intensities - but also directions - which we can use for geometric reconstruction.
:::

# Pinhole Camera Model

![Light passing through a pinhole camera.](assets/svg/pinhole1.svg)

::: notes
I want to introduce the concept of the pinhole camera.
Real pinhole cameras can be made - and work... but here we are describing a model,
that helps us to understand the relationship between points in the world
and points in the image.
:::