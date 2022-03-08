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

## Images as Tensors {data-auto-animate="true}

Images are sampled on a 2D grid.

::: incremental

- Greyscale 2D $h~ \times ~w$
- RGB Images have a 3rd _channel_ dimension.
- Feature images, inside the network, can have many channels.

:::

::: notes
before we step into the design of the network, we need to introduce the notion of a tensor.
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
And this is why we have these particular shaped arrays...
:::

## LeNet for MNIST

![Simplified LeNet](assets/png/lenet.png)

::: notes
definitely simplified - original paper had more layers.
talking through the network
image n*n, filter f*f, >> n - f + 1
:::

## MNIST CNN in PyTorch {data-auto-animate="true"}

```{.python data-line-numbers="1-8|1-3|4-5|6|7-8"}
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 256)
        self.output = nn.Linear(256, 10)
```

::: notes
conv2d: in_channels, out_channels, kernel_size
why 800? 4x4x50 = 800
why 10? 10 classes
:::

## MNIST CNN in PyTorch {data-auto-animate="true"}

```{.python data-line-numbers="1-8|3-4|5|6-8"}
...
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
```

::: notes
and the forward method...
we call the pool method twice - once for each convolutional layer
then we flatten the output - why 800? 4x4x50 = 800
and carry on out with linear layers
:::
