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

## Camera Motion {data-auto-animate="true}

We can obtain:

- 3 **rotation** parameters of the second camera _w.r.t._ the first camera.
- 2 **direction** parameters of the line $B$, connecting the two centres.
- But, we _can't_ estimate the length of $B$.
