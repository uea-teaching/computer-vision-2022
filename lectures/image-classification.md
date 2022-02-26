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

- be invariant to rotation, translation and scaling of an image.
- be invariant to illumination.

::: notes
we don't want the feature to change if it is in a different part of the image.
we don't want the feature to change if it has different illumination...
black cat in coal cellar, white cat in white room, etc.
:::

---

::: columns

::::: column
![texture for features](assets/jpg/texture.jpg)
:::::

::::: column
Texture is a good feature, and often provides good diagnostics.

- e.g. summary statistics on gradient orientations
  :::::

:::
