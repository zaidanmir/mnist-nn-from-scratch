# Backpropagation derivation — 2-layer MLP for MNIST

These notes derive every gradient used by the training loop from first
principles. Each result here corresponds directly to a line in `src/layers.py`
and `src/losses.py`.

## 1. Notation

For a single training example $(x, y)$ with $x \in \mathbb{R}^{784}$ and
$y \in \{0, 1, \ldots, 9\}$:

- $W_1 \in \mathbb{R}^{h \times 784}$, $b_1 \in \mathbb{R}^{h}$ — first layer, with $h$ hidden units (we use $h = 128$).
- $W_2 \in \mathbb{R}^{10 \times h}$, $b_2 \in \mathbb{R}^{10}$ — output layer.
- $\mathbf{1}_y \in \mathbb{R}^{10}$ — one-hot encoding of the true label $y$.

We will derive gradients for one example. Mini-batch gradients are simply
the per-example gradients averaged over the batch (Section 6).

## 2. Forward pass

$$
z_1 = W_1 x + b_1
\qquad
a_1 = \text{ReLU}(z_1) = \max(0, z_1)
$$

$$
z_2 = W_2 a_1 + b_2
\qquad
p = \text{softmax}(z_2), \quad p_k = \frac{e^{z_{2,k}}}{\sum_{j} e^{z_{2,j}}}
$$

The vector $p \in \mathbb{R}^{10}$ is the predicted class distribution.

## 3. Cross-entropy loss

For a one-hot target $\mathbf{1}_y$:

$$
L = -\sum_{k} \mathbf{1}_y[k] \cdot \log p_k = -\log p_y
$$

The negative log-likelihood of the correct class.

## 4. Backward pass — output layer

The well-known clean result for softmax + cross-entropy is:

$$
\frac{\partial L}{\partial z_{2,k}} = p_k - \mathbf{1}_y[k]
$$

Derivation. Start from $L = -\log p_y$ and the softmax definition $p_k = e^{z_{2,k}} / S$ where $S = \sum_j e^{z_{2,j}}$.

$$
\frac{\partial L}{\partial z_{2,k}}
= -\frac{1}{p_y} \cdot \frac{\partial p_y}{\partial z_{2,k}}
$$

The Jacobian of softmax is

$$
\frac{\partial p_y}{\partial z_{2,k}} = p_y(\delta_{yk} - p_k)
$$

(where $\delta_{yk} = 1$ if $y = k$, else $0$). Substituting:

$$
\frac{\partial L}{\partial z_{2,k}}
= -\frac{1}{p_y} \cdot p_y(\delta_{yk} - p_k)
= p_k - \delta_{yk}
= p_k - \mathbf{1}_y[k]
$$

In vector form:

$$
\boxed{\; \delta_2 := \frac{\partial L}{\partial z_2} = p - \mathbf{1}_y \;}
$$

This is why softmax + cross-entropy is paired together — the gradient
collapses to a one-line subtraction, no Jacobian materialised.

The gradients of the output-layer parameters follow from the chain rule
applied to $z_2 = W_2 a_1 + b_2$:

$$
\frac{\partial L}{\partial W_2} = \delta_2 \, a_1^\top
\qquad
\frac{\partial L}{\partial b_2} = \delta_2
$$

## 5. Backward pass — hidden layer

Propagate $\delta_2$ through the linear layer to the activation:

$$
\frac{\partial L}{\partial a_1} = W_2^\top \delta_2
$$

Now through the ReLU. Since $a_1 = \max(0, z_1)$ is applied elementwise,

$$
\frac{\partial a_{1,i}}{\partial z_{1,i}} =
\begin{cases}
1 & z_{1,i} > 0 \\\\
0 & z_{1,i} \le 0
\end{cases}
$$

So:

$$
\delta_1 := \frac{\partial L}{\partial z_1} = (W_2^\top \delta_2) \odot \mathbb{1}[z_1 > 0]
$$

(where $\odot$ is elementwise multiplication). And the input-layer parameter
gradients:

$$
\frac{\partial L}{\partial W_1} = \delta_1 \, x^\top
\qquad
\frac{\partial L}{\partial b_1} = \delta_1
$$

## 6. Mini-batch averaging

For a mini-batch of $B$ examples, the per-batch loss is the mean of
per-example losses, $L_\text{batch} = \frac{1}{B} \sum_{i=1}^{B} L^{(i)}$.
By linearity of differentiation, the batched gradients are the per-example
gradients averaged across the batch. In practice this is one matrix
multiplication: stacking inputs as $X \in \mathbb{R}^{B \times 784}$ and
collected $\delta_2$'s as $\Delta_2 \in \mathbb{R}^{B \times 10}$,

$$
\frac{\partial L_\text{batch}}{\partial W_2} = \frac{1}{B} \, \Delta_2^\top A_1
$$

(and likewise for $W_1$). The implementation works in this batched form
throughout — single-example formulas are only used for derivation.

## 7. SGD update

With learning rate $\eta$, the parameter update for any parameter
$\theta \in \{W_1, b_1, W_2, b_2\}$ is

$$
\theta \leftarrow \theta - \eta \, \frac{\partial L_\text{batch}}{\partial \theta}
$$

That is the entire optimisation rule. Variants like momentum and Adam build
on top of it — pure SGD is what we implement.

## 8. Numerical stability

Two practical issues arise when implementing the above naively in `float32`:

1. **Softmax overflow.** $e^{z_{2,k}}$ blows up if any logit is large.
   Standard fix: subtract the max logit before exponentiating,
   $p_k = e^{z_{2,k} - z_2^\max} / \sum_j e^{z_{2,j} - z_2^\max}$.
   Mathematically equivalent (numerator and denominator scale by the same
   constant); numerically much safer.
2. **$\log 0$.** If softmax outputs a vanishing $p_y$, $-\log p_y$ becomes
   $+\infty$. The fix is to combine softmax + log into `log_softmax` and
   evaluate it in a single numerically stable expression rather than
   composing them naively.

Both are implemented in `src/losses.py` and called out in the docstrings
that reference this section.
