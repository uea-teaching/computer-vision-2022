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

A **camera pair** is two configurations from which images have been taken of the same scene.

::: notes
The definition means we want to estimate something...
often the relative orientation of the two cameras when taking the images

we talk about relative orientation and the two important matrices.
:::

## Orientation {data-auto-animate="true}

The **orientation** of the camera pair can be described using _independent_ orientations for each camera.

How many parameters are needed?

::: notes
These parameters are those involved in x=PX.
So first, how does it work for a calibrated camera?
can anyone remember how many parameters we need for camera extrinsic? Yes - 6 - for the rotation and translation. So 12 for two cameras...

Then for uncalibrated cameras we need the 5 extra values for each camera, so an extra 10 for the pair. A total of 22 parameters.
:::

## Orientation {data-auto-animate="true}

The **orientation** of the camera pair can be described using _independent_ orientations for each camera.

How many parameters are needed?

- _Calibrated_ cameras require **12** parameters.
- _Uncalibrated_ cameras require **22** parameters.

## Camera Motion {data-auto-animate="true}

Can we **estimate** the camera motion without
_knowing_ the scene?

::: notes
so - the question is can we estimate the camera motion without knowing the scene?
previously we needed knowledge of the scene, ie DLT with 6 points in the scene.
Or a camera target with known position and orientation.

Now think about only having images from two cameras, without any knowledge of the scene at all. Which parameters can we obtain from these images?
:::

## Camera Motion {data-auto-animate="true}

Which parameters can be obtained from these images?

- and which cannot?

::: notes
Some parameters we may not be able to estimate at all...
One thing you might have realised is - it is very difficult to estimate scale.

As humans we have learnt that objects belong to a certain size range - but sometimes we can be fooled...

From only images (no knowledge of the scene) - it is impossible to estimate scale in a single image.

:::

## Cameras Measure Direction {data-auto-animate="true}

We can't obtain _global_ **translation** and **rotation** or **scale**.

::: notes
Without knowledge of the scene we can't position the camera or find the heading.
We don't know the scale of the scene.
:::

## Cameras Measure Direction {data-auto-animate="true}

![Two views](assets/svg/two-view.svg){width=80%}

::: notes
let's talk about what's going on here.
first camera and second camera are looking at the same scene.
but if we had a bigger object and moved the second camera - we'd have identical images.

This is important to emphasise...

And it is probably obvious - how could we know where in the world our cameras are?
We cant't estimate the first camera pose - 6 values, and scale...
so 7 parameters we can't estimate!
:::

## Cameras Measure Direction {data-auto-animate="true}

We can obtain:

- 3 **rotation** parameters of the second camera _w.r.t._ the first camera.
- 2 **direction** parameters of the line $B$, connecting the two centres.
- But, we _can't_ estimate the length of $B$.

::: notes
We don't know how far the second camera is only the direction.
We don't know the global position - only with respect to the first camera.
:::

## Calibrated Cameras {data-auto-animate="true}

- We need $2 \times 6 = 12$ parameters for two _calibrated_ cameras for their pose.
- Without additional information we can only obtain $12 - 7 = 5$ parameters.
- Not 3 rotation, 3 translation, and 1 scale.

::: notes
With a calibrated camera, we obtain an angle-preserving model of the object.
We cannot resolve the pose of the first camera - nor the distance between the two cameras.
:::

## Photogrammetric Model {data-auto-animate="true}

Given two cameras images, we can reconstruct an object up to a **similarity** transform.

::: notes
From two images we can construct what is called a **photogrammetric model**.
This is a 3D model of the scene up to a similarity transform - translate - rotate - uniform scale.
We can construct a 3D model, but it will not be aligned or scaled within the 3D world.
:::

## Photogrammetric Model {data-auto-animate="true}

The orientation of the photogrammetric model is called the **absolute** orientation.

- To _obtain_ the absolute orientation we need at least 3 points in 3D.

::: notes
We must distinguish between relative orientation and absolute orientation.
Relative means the second camera with respect to the first...
If we want to align the model to the world - an absolute orientation - we need to get 3 points in 3D to obtain the absolute orientation and those missing 7 parameters.

Conversely, if we know the location of the cameras, we can find the 3D location of point.
:::

## Uncalibrated Cameras {data-auto-animate="true}

For **uncalibrated** cameras, we can only obtain $22-15=7$ parameters given two images.

We need at **least 5 points** in 3D to obtain the absolute orientation.

::: notes
when talking about uncalibrated - I mean only linear errors - not lens distortions, or other non-linear errors.

We are missing the projective transformation in 3D - which has 15 parameters.
A 4x4 matrix - and ignoring the homogeneous scaling.
:::

## Relative Orientation {data-auto-animate="true}

| Camera       | image | pair | RO  | AO  | 3D  |
| :----------- | :---: | :--: | :-: | :-: | :-: |
| Calibrated   |   6   |  12  |  5  |  7  |  3  |
| Uncalibrated |  11   |  22  |  7  | 15  |  3  |

- RO : relative orientation
- AO : absolute orientation
- 3D : minimum number of control points in 3D

::: notes
to summarise the available parameters in an image, in an image pair...
for calibrated every camera has 6 extrinsics... so 12 in total. We can find 5 parameters just from the two images, and the other 7 parameters are missing.
We would need to know 3 points in 3D to find those 7 parameters and thus the absolute orientation.

Similarly for uncalibrated cameras, we need at least 5 points in 3D to obtain the absolute orientation.
:::

## Relative Orientation {data-auto-animate="true}

By simply moving the camera in the scene we can obtain a **relative orientation**.

"Agarwal, Sameer, et al. Building rome in a day. 2011"

![Rome in a day](assets/png/colluseum-2106-photos.png){width=80%}

::: notes
As an inspirational interlude - just by finding image correspondences we can find the relative orientation - and for many images we can reconstruct many points in complex models.
:::
