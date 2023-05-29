r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**
Yes. back-propagation is required in order to train neural networks with decent based optimization.
The main reason is that it enables efficient computation of the gradients for each parameter of the network.
Back-propagation uses the chain-rule to calculate the derivative of the loss w.r.t each parameter efficiently, without it 
it would be very challenging and inefficient to calculate the gradient for each parameter of the network manually.
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
        3e-3,
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
    wstd = 0.5
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. We can see that the no-dropout graph achieves very good training accuracy and low average loss.
But on the other hand, it performs poorly on the test set, with an accuracy of only 22.5 and a relatively high loss.
We see that the dropout graphs have a worse performance on the training set, with higher average loss and much less accuracy compared to the no-dropout case.
But we see that the dropout graphs performed better on the test set and have better generalization.

 This matches what we expected to see since dropout is known to help reduce over-fitting in cases such as ours where the training set is very small.
 Again as can be seen in the dropout graphs, we didn't get a high accuracy on the training data but still managed to achieve better performance 
 on the test set compared to the no-dropout case.
 
 2. We can see from the graphs that the low-dropout model achieves better accuracy on both the training set and the test set.
 We can also deduce from the plots that the high-dropout model tends more to under-fitting.
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible.
The cross-entropy loss compares the predicted class labels to the true labels, where the prediction is usually a softmax probability.
It also penalizes based on how far the predicted probability is from the true label, in other words how sure the model is of its prediction.
The accuracy on the other hand, depends only on whether the prediction was correct or not.
Therefore, it's possible for a few epochs for the loss to increase (since the model is temporarily less sure of its prediction) 
while the accuracy also increases since the models predictions are still above a chosen threshold and are classified correctly.

"""

part2_q3 = r"""
**Your answer:**
1. The difference is that gradient descent is the optimization algorithm used to minimize the loss function during training,
while backpropagation is the specific algorithm used to calculate the gradients necessary for the weight updates during gradient descent.
Backpropagation is an essential component of training neural networks, since it enables efficient computation of gradients.

2. Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are both optimization algorithms used to update the weights of a neural network during training.
The main difference is that in GD the weights are updated by computing the average gradient over the entire training dataset.
While SGD updates the weight based on the gradients calculated with a single example from the dataset.
This leads to several differences in the behavior and performance of the algorithms:
    1. A single step in SGD is more efficient compared to a single step in GD in terms of time and space, since SGD computes the gradients based on individual samples.
    2. In practice, many times SGD converges faster then GD.
    3. In SGD, the path in the parameter space is usually more erratic and noisy due to the randomness introduced by the selection of examples.

3. SGD is widely used in the practice of deep learning for several reasons:
    1. Computational Efficiency: SGD is computationally more efficient compared to Gradient Descent (GD) as it updates the model's parameters based on individual training examples or small mini-batches.
    2. Convergence Speed: In practice, SGD converges faster then GD in many cases.
    3. Regularization and Generalization: SGD, with its stochastic nature, introduces noise into the parameter updates. This noise can act as a form of regularization, preventing over-fitting and improving generalization performance.
    4. Avoiding Local Minima: Deep learning models often have complex loss landscapes with many local minima.
     SGD's stochastic updates can help the optimization process by allowing the algorithm to jump out of shallow local minima and explore different regions of the parameter space.
     This increased exploration ability makes SGD less likely to get stuck in suboptimal solutions compared to GD. 
    5. Online Learning: In scenarios where data arrives in a streaming or online fashion, SGD is well-suited. 
        It can efficiently update the model with new examples as they become available.
        
4. 1. This approach will produce a gradient equivalent to GD since a gradient is a linear operator.
    Meaning that the sum of gradients is equal to the gradient of a sum, mathematically: $ \nabla_\theta\left[\sum L(\theta,X)\right] = \sum \nabla_\theta L(\theta,X) $
    <br> This is correct since the batches we use are disjoint and the union of the batches is the entire dataset.
    2. Despite using smaller batch sizes to fit the data in memory, an out of memory error occurred because the intermediate data required for the backward pass still exceeded the available memory.
"""

part2_q4 = r"""
**Your answer:**
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
1. As we can see the training loss is very close to zero, therefore we would say that 
our model doesn't suffer from an high optimization error.

2. As we can see our first binary classifier achieves a relatively high accuracy (above 85%), therefore we would say that 
it doesn't suffer from an high generalization error.

3. The generalization error is the sum of the approximation error and estimation error 
(estimation error meaning how good is our hypotheses compared to the optimal hypotheses from our hypotheses class).
Therefore, if the generalization error is low as we argued before, then the approximation error cannot be high as well.

"""

part3_q2 = r"""
**Your answer:**
We now that the training data is rotated by 10 degrees and the validation data is rotated by 50 degrees (counter-clockwise), therefore on the validation set we would expect 
more points from class 1 (orange points) to be misclassified compared to the number of points from class 0 (blue points) which are misclassified.
This is because the decision boundary of out classifier is based on the training data and due to the rotation of the validation data, more orange points will cross to the wrong side of the decision boundry,
in comparison to the blue points.

In other words, we would expect the number of TP (true positive) data points to decrease in comparison to the number of TN (true negative) points. <br>
So, the TNR will be higher then the TPR.<br>
And therefore we expect the FNR = 1 - TPR to be higher then the FPR = 1 - TNR, which is exactly what happens in our case.<br>
From the first confusion matrix we have:<br>
FNR = $\frac{{FN}}{{P}} = \frac{{0.13}}{{0.13 + 0.37}} = 0.26$<br>
FPR =  $\frac{{FP}}{{N}} = \frac{{0.0086}}{{0.0086 + 0.49}} = 0.017$<br>
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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