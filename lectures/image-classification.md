---
title: Image Classification
subtitle: Computer Vision CMP-6035B
author: Dr. David Greenwood
date: March 2022
---

# Content

- HOG features
- Visual Words
- Spatial Pyramid
- PCA and LDA
- Evaluation

::: notes
briefly discuss HOGs...
some other classification methods...
PCA and LDA - PCA is everywhere!
Some more on evaluation...

This week we will look at two computer vision application scenarios which can employ the aforementioned classifiers.

:::

# Image Classification

Passing a **whole** image to a classifier.

::: notes
We have previously studied various classifiers.
the discussion talked of clouds of points to be classified, and these points will image features.
:::

# Feature Extraction

What are good features?

## Feature Extraction {data-auto-animate="true"}

The main difficulty in solving these image classification problems is finding good image features.

::: notes
the trick is to extract good features... but what are they?
:::

## What are good features? {data-auto-animate="true"}

::: incremental

- Good features should **exhibit** _between-class_ variation.
- Good features should **suppress** _within-class_ variation.

:::

::: notes
to classify cats and dogs, we want features extracted from cats to be very different from features extracted from dogs. - between-class variation

also, we want features extracted from different cats to be a s similar as possible.
:::

---

Other desirable properties of features are:

::: incremental

- _invariant_ to rotation, translation and scaling of an image
- _invariant_ to illumination

:::

::: notes
we don't want the feature to change if it is in a different part of the image.
we don't want the feature to change if it has different illumination...
black cat in coal cellar, white cat in white room, etc.
:::

## What are good features? {data-auto-animate="true"}

::: columns

::::: {.column width=40%}
![texture for features](assets/jpg/texture.jpg){width="75%"}
:::::

::::: column

Texture is a good feature, and often provides good diagnostics.

- e.g. summary statistics on gradient orientations

:::::

:::

::: notes
you might guess - texture is a good feature...
we learned earlier how to calculate gradient and gradient direction...
gradient captures the smallest features of an image...
:::

---

::: columns
::::: {.column width=35%}

![kitchen 1](assets/jpg/sun_aaesgnhzvszupuvo.jpg){width="85%"}

![kitchen 2](assets/jpg/sun_aaevfnfhjudhbvxh.jpg){width="85%"}

:::::

::::: column

Exact feature locations are not important.

- Small variations in the layout will not change the class label.

:::::

:::

::: notes
These images from the coursework...
15 classes - one of which is kitchen...
it does not matter where in the image is a window or microwave...
translation invariance is important...
:::

## Classification Applications

Classify an X-ray image as containing cancer or not.

::: incremental

- A _binary_ classification problem.
- Normally requires significant human expertise!

:::

::: notes
We could fill the entire lecture with examples of classification applications, such is the rapid expansion of the field.
:::

---

Material classification, eg. wood, metal, plastic, etc.

::: incremental

- Texture is likely useful, but...
- Illumination may significantly change the texture.
- Extract features invariant to illumination.

:::

---

Scene classification e.g. kitchen, bathroom, beach.

::: incremental

- Importance of context.
- Scenes contain many objects, but their exact location is less important.

:::

# Image Classification Strategies

Extracting _low level_ features from an image.

::: notes
usually we would extract some low level features...
:::

## Low Level Features

The low level features, which are used the most often include
_SIFT_ and _HOG_ features combined with some colour descriptors.

---

SIFT - Scale Invariant Feature Transform

::: incremental

- Localised feature based on image gradients.
- One of the first of its kind.
- Some proprietary aspects to its use.
- covered in a later lecture.

:::

---

HOG - histograms of oriented gradients.

- Also a gradient based feature.
- next up!

## Histograms of Oriented Gradients {data-auto-animate="true"}

- Image is divided into regions - a window.
- Each window is further divided into cells.
- Each cell is typically 6 to 8 pixels wide.

::: notes
In the paper they describe the window as containing a grid of cells.
so in practice, we decide a cell size, the build windows of cell blocks.
:::

## Histograms of Oriented Gradients {data-auto-animate="true"}

A local 1D histogram of _gradient_ directions.

::: incremental

- 1D dimension is the **angle** of the gradient
- the angle is _quantised_ into a discrete set of bins
- for example, for a bin size 20 degrees, we have 18 bins
- sum of all elements is equal to number of pixels in the _cell_

:::

::: notes
so far you have been working with histograms of colour values.
:::

## Angle {data-transition="slide"}

- A gradient is calculated using a centred $[-1,0,1]$ filter.
- The filter is applied vertically and horizontally.
- We derive the gradient direction from these first derivatives.

$$\alpha = \tan^{-1} \frac{\delta g}{\delta y}~ / ~ \frac{\delta g}{\delta x}$$

## Magnitude {data-transition="slide"}

For colour images, we can calculate gradient for the three channels and select the one with the largest _magnitude_.

$$|G| = \sqrt{\left(\frac{\delta g}{\delta x}\right)^2 + \left(\frac{\delta g}{\delta y}\right)^2} $$

::: notes
and, of course, we can get the gradient magnitude for each pixel with Pythagoras.
:::

## Binning {data-transition="slide"}

For each pixel within a cell, its gradient _orientation_ is used to increment the relevant histogram bin.

::: incremental

- in _proportion_ to the gradient magnitude

:::

::: notes
different to colour histogram, where we just increment on value of colour.
this is to promote edges in the image
:::

## Interpolation {data-transition="slide"}

To enforce invariance to some small gradient orientation differences, we _interpolate_ histogram contributions between the neighbouring bin centres.

::: incremental

- Typical binning - 20 degrees.

:::

::: notes
to avoid bin mis-match where a small variation in a gradient can shift the assignment of a pixel to a bin.
:::

## Contrast Normalisation {data-transition="slide"}

We choose a certain configuration of cells and call it a _block_

::: incremental

- typically 2-3 cell wide
- perform _normalisation_ within each block
- various schemes proposed in original paper
- e.g. modified L2 norm $v \rightarrow v / \sqrt{||v||^2_2 + \epsilon^2}$

:::

::: notes
this step imparts some illumination invariance - the epsilon is a small constant to avoid division by zero.
:::

## {data-transition="slide"}

![HOG example](assets/png/hog_example.png){width="85%"}

Dalal and Triggs. "Histograms of Oriented Gradients for Human Detection", CVPR, 2005

::: notes
Histogram features from all cells are combined forming
a feature vector which can be used for classification.
We will look at this paper again later today when discussing applications.
A very influential paper - 40k citatations.
:::

# Visual Words {data-transition="convex"}

Once the features are extracted, we would often use _dictionaries_ of **visual words**.

::: notes
we could use the features directly - but it is better to do something extra...
:::

# Visual Words {data-auto-animate="true"}

Features representing scenes should be able to **summarise** these scenes.

# Visual Words {data-auto-animate="true"}

Imagine we would like to classify images containing _sets_ of objects.

::: notes
rather like your coursework - a kitchen has a lot of objects common to other kitchens.
:::

# Visual Words {data-auto-animate="true"}

The precise location of objects may not be relevant.

::: incremental

- The objects may move or deform within the image.
- The viewpoint may change or the image may be deformed or scaled.

:::

::: notes
rather like your coursework - a kitchen has a lot of objects common to other kitchens.
Some objects may not be present in some scenes.
:::

# Visual Words {data-auto-animate="true"}

This suggests some kind of high level histogram representation of the scene.

::: incremental

- How many cups or plates visible in a kitchen scene?
- Will these objects be present in an outdoor scene?
- How many trees might you expect in a kitchen?

:::

::: notes
Think of high level features as being the bins of a histogram.
Of course HOG do not directly represent these objects...but edges , corners etc..
:::

# Visual Words {data-auto-animate="true"}

Detect _interest_ points in the image.

::: incremental

- e.g. corners, T-junctions etc.
- build _neighbourhoods_ around them.

:::

::: notes
How do we achieve this high level histogram?
:::
