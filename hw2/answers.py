r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1. the Jacobian tensor would have the shape $512 \times 64 \times 1024 \times 64$. Each entry of the tensor corresponds to the partial derivative $\frac{\partial Y_{i,j}}{\partial X_{n,m}}$, where $i$ ranges from 1 to 512, $j$ ranges from 1 to 64, $n$ ranges from 1 to 1024, and $m$ ranges from 1 to 64. 
2. We model the formula as $Y = W^{T} \cdot X$, where $W \in \mathbb{R}^{1024 \times 512}$ and $X \in \mathbb{R}^{1024 \times 64}$.
The Jacobian tensor is the $\frac{\partial Y_{i,j}}{\partial X_{n,m}}$. However, $Y_{i,j}$ is equal to the dot product of row $i$ of $W$ and column $j$ of $X$. This means that $Y_{i,j}$ is not dependent on the entries of $X$ that are not in column $j$, and therefore it is sparse.
3. No, we do not need to materialize the Jacobian tensor in order to calculate the downstream gradient $\delta\mat{X}$ without explicitly forming the Jacobian. Instead, we can directly compute it using the chain rule.
Given the gradient of the output with respect to the scalar loss, $\delta\mat{Y} = \frac{\partial L}{\partial \mat{Y}}$, we can calculate the downstream gradient $\delta\mat{X}$ as follows:
$\delta\mat{X} = \frac{\partial L}{\partial \mat{X}} = \delta\mat{Y} \cdot W^\top$
Here, $\delta\mat{Y}$ is the given gradient and $W^\top$ represents the transpose of the weight matrix $W$. This computation avoids the need to explicitly materialize the Jacobian tensor and is more computationally efficient.



"""

part1_q2 = r"""
**Your answer:**

Backpropagation is crucial for training neural networks efficiently. Compared to a naive Jacobian-based method that computes gradients, backpropagation significantly reduces the number of computation steps required to calculate gradients in deep neural networks. Backpropagation achieves computational savings by efficiently reusing intermediate results, and avoiding unnecessary calculations as we saw in question 1, resulting in a more efficient and scalable approach for gradient computation
For instance, consider a deep neural network with 100 layers. In a naive Jacobian-based method, to calculate the gradients of the loss function with respect to the input, we would need to compute the full Jacobian matrix. This matrix would have dimensions that grow exponentially with the number of layers, making the computation infeasible.


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    wstd, lr, reg = 0.1, 10e-2, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.05,
        3e-2,
        8e-3,
        0,
        1e-3,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 8
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 5e-3
    weight_decay = 1e-2
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    weight_decay = 1e-2
    momentum = 0.9


    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model performance was not good, in the first image it detected a surfboard which doesn't exist in the image and misclsified the two dolphines as persons.
In the second image we can say that the performance was slightly better but still not good enough. 
In terms of segmentation it performed poorly on the cat+dog in the left side of the image(which were segmented as the same object), but the classification was poor
it misclassified two of the dogs as cats (one of them is segmented with the cat but the dog still takes most of the bounding box). 
2. In the first image, we can see that the model doesn't cotain a "dolphin" class, therefore it classified the dolphins as persons.
In the second image, the objects in the image have a high overlap and are located very close to each other. Therefore, the model had a hard time to classify and segment the objects.
"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**
1. In the first image (donuts) we can see that the model detected 3 donuts out of 5 possible, and it may be because the bluriness of the image, and the fact that the two other donuts are cropped.
<br> In the second image we can see that the model did not detect the person in the image, and it may be because of the poor lighting in the image.
<br> In the third image we can see that out of many possible forks, spoons and knives, the model detected only a single spoon, which demonstrating the affect of clutterness on the model performance. 

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""