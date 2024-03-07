# On the Diminishing Returns of Width for Continual Learning

This repository contains the code to reproduce the empirical results in the paper [On the Diminishing Returns of Width for Continual Learning](Link). 


## Summary of Results

### Abstract

While deep neural networks have demonstrated groundbreaking performance in various settings, these models often suffer from catastrophic forgetting when trained on new tasks in sequence. Several works have empirically demonstrated that increasing the width of a neural network leads to a decrease in catastrophic forgetting but have yet to characterize the exact relationship between width and continual learning. In this paper. we design one of the first frameworks to analyze Continual Learning Theory and prove that width is directly related to forgetting in Feed-Forward Networks (FFN). In particular, we demonstrate that increasing network widths to reduce forgetting yields diminishing returns. We empirically verify our claims at widths hitherto unexplored in prior studies where the diminishing returns are clearly observed as predicted by our theory.

### Theoretical Contributions

Our results contribute to the literature examining the relationship between neural network architectures and continual
learning performance. We provide one of the first theoretical frameworks for analyzing catastrophic forgetting in Feed-Forward Networks. While our theoretical framework does not perfectly capture all information about continual forgetting empirically, it is a valuable step in analyzing continual learning from a theoretical framework. As predicted by our theoretical framework, we demonstrate empirically that scaling width alone is insufficient for mitigating the effects
of catastrophic forgetting, providing a more nuanced understanding of finite-width forgetting dynamics than results achieved in prior studies. 

### Empirical Validation

Rotated MNIST (1 Layer MLP)

| Width | AA   | AF   | LA   | JA   |
|--------------------|------|------|------|------|
| 32     | 56.3 | 37.7 | 93.0 | 91.8 |
| 64    | 58.7 | 36.0 | 93.5 | 93.5 |
| 128  | 59.8 | 35.0 | 93.8 | 94.3 |
| 256   | 60.9 | 34.2 | 94.0 | 94.8 |
| 512   | 61.9 | 33.2 | 94.1 | 95.0 |
| 1024  | 62.7 | 32.6 | 94.2 | 95.3 |
| 2048  | 64.1 | 31.2 | 94.3 | 95.5 |
| 4096  | 65.3 | 30.2 | 94.5 | 95.7 |
| 8192  | 66.7 | 28.9 | 94.7 | 95.7 |
| 16384   | 68.0 | 27.9 | 94.9 | 95.9 |
| 32768  | 69.4 | 26.6 | 95.6 | 96.1 |
| 65536  | 69.6 | 26.7 | 95.6 | 96.2 |

Rotated Fashion MNIST (1 Layer MLP)

| Width (Parameters) | AA   | AF   | LA   | JA   |
|--------------------|------|------|------|------|
| 32      | 37.7 | 46.0 | 82.1 | 77.8 |
| 64      | 37.9 | 46.0 | 82.4 | 80.0 |
| 128     | 38.2 | 46.0 | 82.5 | 79.4 |
| 256     | 38.4 | 45.9 | 82.7 | 79.8 |
| 512     | 38.8 | 45.6 | 82.9 | 79.9 |
| 1024   | 39.3 | 45.3 | 83.1 | 79.9 |
| 2048 | 39.9 | 44.8 | 83.3 | 79.1 |
| 4096 | 40.1 | 44.9 | 83.7 | 80.9 |
| 8192 | 40.8 | 44.5 | 83.9 | 80.2 |
| 16384   | 41.4 | 44.3 | 84.5 | 78.8 |
| 32768    | 41.9 | 44.3 | 84.9 | 79.9 |
| 65536    | 42.0 | 44.6 | 85.5 | 80.9 |




## Getting Started

To begin, first clone the repository. The only external dependencies needed to run the code are PyTorch and
torchvision, which can be installed following the instructions on the [PyTorch website](https://pytorch.org/)


## Running Experiments

To reproduce any of the experiments from the paper, execute the `run.py` script and specify the task name, depth, and the layer width. For example

```
python3 run.py --task_name mnist --num_layers 2 --width 1024
```

```
python3 run.py --task_name svhn --num_layers 1 --width 64
```

```
python3 run.py --task_name gtsrb --num_layers 3 --width 256
```

## Citation