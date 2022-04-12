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

# Pinhole Camera Model {data-auto-animate="true"}

![Light passing through a pinhole camera.](assets/svg/pinhole1.svg)

::: notes
I want to introduce the concept of the pinhole camera.
Real pinhole cameras can be made - and work... but here we are describing a model,
that helps us to understand the relationship between points in the world
and points in the image.
:::

## {data-auto-animate="true"}

- $f$ : effective focal length
- $\textbf{r}_{o} = (x_o, y_o, z_o)$
- $\textbf{r}_{i} = (x_i, y_i, f)$

![Camera at the origin.](assets/svg/pinhole2.svg)

::: notes
Camera position is often called principal point.
The distance from the principal point to the image plane is called the focal length.
We can look at a ray from an object to the principal point r_o,
that then passes through the pinhole to the image plane r_i.

NB. The pinhole diagram is often shown with the image plane in front of the camera.
NB2. Different texts apply different labels.
:::

## Pinhole Camera Model {data-auto-animate="true"}

Using similar triangles, we get the equations of perspective projection.

$$
\frac{\textbf{r}_{i}}{f} = \frac{\textbf{r}_{o}}{z_o} \quad \Rightarrow \quad
\frac{x_i}{f} = \frac{x_o}{z_o}, ~\frac{y_i}{f} = \frac{y_o}{z_o}
$$

::: notes
The 2d x, y on the image plane are derived by dividing the 3D by z, scaled by the focal length.
very simple equations - but can produce some very unintuitive effects.
:::

# Camera Parameters {data-auto-animate="true"}

Describe how a world point is mapped to a pixel coordinate.

::: notes
Our goal is to describe how a point in the world maps to a pixel in the image.
:::

## Camera Parameters {data-auto-animate="true"}

Describe how a world point is mapped to a pixel coordinate.

![point mapping](assets/svg/parameters1.svg)

## Camera Parameters {data-auto-animate="true"}

We will describe this mapping in **homogeneous** coordinates.

::: {style="font-size:1.5em"}

$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} =
P \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

:::

## Coordinate Systems {data-auto-animate="true"}

We have to transform via a number of coordinate systems:

::: incremental

- The world coordinate system
- The camera coordinate system
- The image coordinate system
- The pixel coordinate system

:::

## World to Pixels {data-auto-animate="true"}

![World to Pixels](assets/svg/world-to-sensor1.svg)

::: notes
we start with a point in the world on the left of the image.
and move to the right in the diagram giving us the final pixel coordinate.
:::

## World to Pixels {data-auto-animate="true"}

![World to Camera coordinates](assets/svg/world-to-sensor2.svg)

::: notes
Object to camera is in 3D and is invertible.
We convert from one coordinate frame to another.
:::

## World to Pixels {data-auto-animate="true"}

![Projection to 2D](assets/svg/world-to-sensor3.svg)

::: notes
Then we project from 3D to 2D.
This is not invertible - we lose some information here.
:::

## World to Pixels {data-auto-animate="true"}

![Convert to Sensor coordinates](assets/svg/world-to-sensor4.svg)

::: notes
The sensor coordinates need to be transformed from the projected coordinates.
Usually we have 0,0 in the top left of an image, whereas we have projected to the centre.
:::

## World to Pixels {data-auto-animate="true"}

![Lens Distortions](assets/svg/world-to-sensor5.svg)

::: notes
In the real world, lenses have distortions and aberrations that distort the image.
We can correct for these here.
:::

## Camera Parameters {data-auto-animate="true"}

How do we work with these parameters?

::: incremental

- _extrinsic_ parameters: the pose of the camera in the world
- _intrinsic_ parameters: the properties of the camera

:::

::: notes
we form two groups of parameters:
intrinsic - the ideal projection to 2D and then translation to pixel coordinates
extrinsic - the pose of the camera in the world
you can imagine if you pick up your camera and move it - it does not effect the position of the sensor, the pixel shape, the focal length, etc.
:::

## Extrinsic Parameters {data-auto-animate="true"}

- Describe the **pose**, the _position_ and _heading_, of the camera in the world.
- Invertible transformation.

How many parameters do we need?

::: incremental

- 3 parameters for the position
- 3 parameters for the heading
- There are **6** _extrinsic_ parameters.

:::

::: notes
the only thing we can do is translate, and rotate the camera.
:::

## Extrinsic Parameters {data-auto-animate="true"}

Point in world coordinates:

$$
\textbf{\textit{X}}_p = [ \textit{X}_p, \textit{Y}_p, \textit{Z}_p ]^T
$$

Origin of camera in world coordinates:

$$
\textbf{\textit{X}}_o = [ \textit{X}_o, \textit{Y}_o, \textit{Z}_o ]^T
$$

## Transformation {data-auto-animate="true"}

**Translation** between origin of world and camera coordinates is:

$$
\textbf{\textit{X}}_o = [ \textit{X}_o, \textit{Y}_o, \textit{Z}_o ]^T
$$

**Rotation** $R$ from world to camera coordinates system is:

$$
{}^{k}\textbf{\textit{X}}_p = R(\textbf{\textit{X}}_p - \textbf{\textit{X}}_o)
$$

## Homogeneous Coordinates {data-auto-animate="true"}

$$
\begin{aligned}
\begin{bmatrix} {}^{k}\textbf{\textit{X}}_p \\ 1 \end{bmatrix} &=
\begin{bmatrix} R \hfill ~\textbf{0} \\  \textbf{0}^T \hfill ~1 \end{bmatrix}
\begin{bmatrix} I_3 \quad -\textbf{\textit{X}}_o \\  \textbf{0}^T \hfill ~1 \end{bmatrix}
\begin{bmatrix} \textbf{\textit{X}}_p \\ 1 \end{bmatrix} \\
&= \begin{bmatrix} R \quad -R \textbf{\textit{X}}_o \\
    \textbf{0}^T \hfill ~1 \end{bmatrix}
\begin{bmatrix} \textbf{\textit{X}}_p \\ 1 \end{bmatrix}
\end{aligned}
$$

or:

$$
{}^{k}\textbf{X}_p  = {}^{k}H \textbf{X}_p
\quad \text{where} \quad
{}^{k}H = \begin{bmatrix} R \quad -R \textbf{\textit{X}}_o \\
    \textbf{0}^T \hfill ~1 \end{bmatrix}
$$

::: notes
here, the bold zero is the zero vector.
R is a 3x3 rotation matrix.
I_3 is the 3x3 identity matrix.
and we can premultiply the rotation and translation because we are in homogeneous coordinates.
So finally, we have H is the extrinsic parameters.
NB Homogeneous coordinates are shown in non-italic font.
:::
