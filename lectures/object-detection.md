---
title: Object Detection
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: March 2022
---

# Content

- Classification or Object Detection
- Sliding Window
- Detecting Faces
- Detecting Humans

# Object Detection

What is object detection?

::: notes
We have just been talking about image classification - what are the differences between classification and object detection?
:::

## Object Detection {data-auto-animate="true"}

_Image classification_ methods can detect an object in the image if there is just a **single** object in the scene and it clearly dominates the image.

If this constraint is not met, we are in the **object detection** scenario.

## Object Detection {data-auto-animate="true"}

We can use similar techniques we have learnt in Image Classification to detect objects in an image.

Here we apply these techniques to **sub-windows** of the image.

# Sliding Window

Sliding window is a _meta_ algorithm - a concept found in many machine learning algorithms.

## Sliding Window
