---
title: Two-View Geometry
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: Spring 2022
---

# Contents {data-transition="convex"}

- Camera Pair
- Essential and Fundamental Matrix
- Epipolar Geometry

# Camera Pair

Two cameras capturing images of the same scene.

::: notes
so far we have talked about the camera model - and we have confined
our discussion to a single camera, and it's calibration.
Now it's time to consider camera pairs.

:::

## Camera Pair {data-auto-animate="true}

![A stereo camera. Intel D435](assets/jpg/intel-d435.jpg){width=80%}

::: notes
this is a camera pair - a stereo camera. left cam, ir-projector, right cam, rgb-cam.
But this is not the only stereo pair.
This camera makes it easy to work with images taken from different locations
at the same time.
This is an example of what we call the stereo-normal case
where both cameras are point in the same direction.

We can equally have two separate cameras,
in two different positions, and take images at the same time.

Or, we could have one camera take an image then move to a different position and take another image.
:::

## Camera Pair {data-auto-animate="true}

- A stereo camera.
- Two cameras, each with a different position.
- One camera that moves.

A camera pair is two configurations from which images have been taken of the same scene.

::: notes
The definition means we want to estimate something...
often the relative orientation of the two cameras when taking the images

we talk about relative orientation and the two important matrices.
:::
