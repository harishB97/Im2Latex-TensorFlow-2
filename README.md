# Im2Latex-TensorFlow-2
TensorFlow-2 implementation of Im2Latex deep learning model described in HarvardNLP paper **"[What You Get Is What You See:
A Visual Markup Decompiler]([https://arxiv.org/abs/1609.04938](https://arxiv.org/pdf/1609.04938v1.pdf))"** which can latex code from images of mathematical expressions 

The model has been built referring the original implementation from **[here](https://github.com/harvardnlp/im2markup)** and TensorFlow-1 implementation from **[here](https://github.com/aspnetcs/myim2latex-tensorflow-docker)**.

Model building, training as well as validation has been included in the notebook "Im2Markup_tf2.ipynb"

Download the tfrecord files from **[here](https://drive.google.com/drive/folders/1eQ3qvM6wpvsL4XbREGfVfJhybDWjUapH?usp=sharing)** and move to "100K_tfrecords_v3" folder before training.

### Preferred versions (python packages):
---

- Tensorflow => 2.4.1
- Numpy => 1.19.5
