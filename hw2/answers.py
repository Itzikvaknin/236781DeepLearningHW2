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
1. In this case, since the patients immediately develop non-lethal symptoms and the treatment isn't expensive,
we would rather have a low as possible FPR (because the diagnosis is expensive) even if it means a higher FNR.
Therefore, we would rather use the previous default threshold (or find some other value on the curve) because according 
to the confusion matrices it will lead to a lower FPR compared to the "optimal" point on the curve.

2. In this case we care more about sensitivity, and will prefer a low as possible FNR.
But we also want a low FPR since the diagnosis is very expensive.
Therefore we will want to use the optimal threshold on the roc curve since it minimizes both FPR and FNR.

"""


part3_q4 = r"""
**Your answer:**
1. With a depth of 1 we can observe that the increasing width doesn't have much affect and the model learns an almost linear 
decision boundary.
 With a depth of two we can see that a higher width creates a less linear decision boundary and increases the test accuracy.
 With a depth of 4, we again see that increasing the width creates a non linear decision boundary, but increasing the width too much
 starts to decrease the test accuracy.
 <br>
 2. With a width of 2 we see that the decision boundary is almost linear and we get a poor test performance.
  With width = 8, 32 we see that increasing the depth creates a less linear decision boundary and generally tends to improve test accuracy.
  <br>
  3. We see that the performance in the case of depth = 4, width = 8 is better then in the case: depth = 1, width = 32.
  In addition, the learned decision boundary with depth = 1 is almost linear despite the model being very wide.
  This may be explained by the tendency of depth to be more important when dealing with MLP models.
  <br>
  4. We can see from the confusion matrices that after the threshold selction
  the TPR and TNR are roughly the same and the FNR and FPR are roughly the same.
  We can also notice that the FNR decreased while the FPR increased.
  The selection had a positive effect on the test accuracy, before the selection we had 
  an accuracy of 86.5% and afterwards and accuracy of 90.8% (with depth = 4, width = 8)
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
In general, depth enriches the model class complexity.
On one hand, higher depth (i.e higher model complexity) has more chance to better fit the underlying distribution.
On the other hand, a too deep network might suffer from conditions such as over-fitting, vanishing gradients and make optimization harder
(since the parameter space becomes more non-convex), which may lead to worse generalization. 
We can see that for experiment 1.1 for both k= 32 and k = 64 the network with L=4 achieved better test and train accuracy, compared to all others.
In addition, we see that in all cases except for the test loss for k = 32, the L=4 network had the lowest loss.
Moreover, for depths L = 8 and L = 16 the networks were not trainable. We believe that this might be due to vanishing gradients, 
which will also explain the non changing loss for these networks.

The issue of vanishing gradients can be solved using the two following methods:
1.Adding skip connections, allowing gradients to flow directly across different layers and preventing them from vanishing during backpropagation.

2.Batch Normalization: Introduce batch normalization layers to normalize the activations within each mini-batch.
"""

part5_q2 = r"""
**Your answer:**
In this experiment, for L=2 the network with K=32 achieved the best test accuracy, though not by a very large amount.
On the other hand, for L=4 we see that a higher number of filters per layer led to better test accuracy, we can see that networks with
K= 64 or K = 128 achieved around 5% more test accuracy compared to K = 32.
For L = 8 we again see that the network was not trainable, which reassures our assumption from experiment 1.1 that 
the depth is what causes the un-trainability of the network rather then the number of filters. 
(This can also be due to a non-optimal choice of hyperparameter from our side, and maybe there is a combination of hyperparameters 
which will make the deeper networks trainable as well).

"""

part5_q3 = r"""
**Your answer:**
We see that the highest test accuracy (in this experiment and also compared to 1.1 and 1.2) is for the network with L=3,
achieving close to 80% accuracy.
We also see that the network with L = 4, was not trainable, this due to the network having in practice 8 convolutional layers,
which as we've seen in previous experiments can be due to the problem of vanishing gradients in deep neural networks.
The better performance compared to experiments 1.1 and 1.2 can be due to the usage of a varying number of filters 
in the convolutional layers, and in particular the last 3 layers are wider then the first 3.
Like we have seen in the lectures, in real world CNN models the deeper convolutional layers often are wider (more filters).
In this experiment we also changed the pool_every parameter to 2 instead of 3 like in previous experiments since from our testing this value produced better results.
"""

part5_q4 = r"""
**Your answer:**
We see that all models trained in this experiment have a better generalization compared to experiment 1.1
But compared to experiment 1.3 they perform worse or the same. 
The major noticeable thing in this experiment in our opinion is the fact that previous un-trainable deep networks are now trainable.
We believe that due to this, and the fact that in general deeper networks tend to have better generalization capabilities, with a more careful
decision of hypermarkets (and maybe even architecture) we could outperform the results from 1.3.
For this experiment we chose to again change the pool_every parameter, int the first configuration we used pool_every = 4 and 
for the second configuration we used pool_every = 2. We also made the the fully connected part wider and deeper which from out 
test runs had made the models to have better test accuracy.
"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model performance was not good, in the first image it detected a surfboard which doesn't exist in the image and misclassified the two dolphins as persons.
In the second image we can say that the performance was slightly better but still not good enough. 
In terms of segmentation it performed poorly on the cat+dog in the left side of the image(which were segmented as the same object), but the classification was poor
it misclassified two of the dogs as cats (one of them is segmented with the cat but the dog still takes most of the bounding box). 
2. In the first image, we can see that the model doesn't contain a "dolphin" class, therefore it classified the dolphins as persons.

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