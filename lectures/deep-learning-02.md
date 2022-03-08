---
title: Practical Deep Learning
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: Spring 2022
---

# Content

- Convolutional Neural Network (CNN)
- Transfer Learning
- Tricks of the Trade
- Work in the Field

# Convolutional Neural Network (CNN)

A simplified LeNet for MNIST digits.

- Gradient Based Learning Applied to Document Recognition. LeCun, et al. 1998

## LeNet for MNIST

![Simplified LeNet](assets/png/lenet.png)

::: notes
talking through the network
image n*n, filter f*f, >> n - f + 1
:::

## Images as Tensors {data-auto-animate="true}

Images are sampled on a 2D grid.

::: incremental

- Greyscale 2D $h~ \times ~w$
- RGB Images have a 3rd _channel_ dimension.
- Feature images, inside the network, can have many channels.

:::

## Images as Tensors {data-auto-animate="true}

In Pytorch, the channel dimension is **before** the spatial dimensions.

$$C~ \times ~H~ \times ~W$$

::: notes
This is not true for all frameworks - and can cause confusion!
:::

## Images as Tensors {data-auto-animate="true}

When training Neural Networks, we use mini-batches.

$$S~ \times ~C~ \times ~H~ \times ~W$$

Hence, we pass **4D** Tensors to the network.

::: notes

:::
