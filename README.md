# Im2Latex-TensorFlow-2
---
TensorFlow-2 implementation of Im2Latex deep learning model for latex code generation from images of mathematical expressions described in HarvardNLP paper **["What You Get Is What You See: A Visual Markup Decompiler"](http://arxiv.org/pdf/1609.04938v1.pdf)**

    What You Get Is What You See: A Visual Markup Decompiler  
    Yuntian Deng, Anssi Kanervisto, and Alexander M. Rush
    http://arxiv.org/pdf/1609.04938v1.pdf

This is a general-purpose, deep learning-based system to decompile an image into presentational markup. For example, we can infer the LaTeX or HTML source from a rendered image.

<p align="center"><img src="media\architecture.png" width="40%" height="40%"></p>

## Training data
---
Source **[im2latex-100k dataset](https://zenodo.org/record/56198#.Ys2HInZBy3B)** has been preprocessed and resized as suitable for the model. 
Download the data from **[this link](https://drive.google.com/file/d/18JW6Dn0M1T_YiANeMfM14tjXMnAXxt-l/view?usp=sharing)** and move to "images" folder before training.

## Sample results
---
<img src="media\result22.jpg" width="60%" height="60%" align="center">
<img src="media\result21.jpg" width="60%" height="60%" align="center">

## Model Performance
---
### BLEU score
1. Validation dataset (10340 images): 84.44%
2. Test dataset (9340 images): 84.30%

### Validation and train perplexity
<img src="media\perplexity_3.jpg" width="80%" height="80%" align="center">

### Exact match accuracy
<img src="media\train_acc_tb_2.jpg" width="80%" height="80%" align="center">

### Preferred versions:
---
- Tensorflow 2.8.0
- Numpy 1.21.6

### Previous implementations:
---
1. **[Original implementation by HarvardNLP in Torch (Lua)](https://github.com/harvardnlp/im2markup)**
2. **[TensorFlow-1 implementation](https://github.com/ritheshkumar95/im2latex-tensorflow)**
