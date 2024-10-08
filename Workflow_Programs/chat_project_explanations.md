In neural networks, scaling (like the use of `StandardScaler`) is often essential due to how neural networks process data. Scaling ensures
that the inputs and outputs are normalized, which helps with several aspects:

1. **Gradient Descent Stability:** Neural networks use gradient descent to adjust weights during training. If the input values vary
   significantly in scale, the gradients can become very large or small, leading to unstable or inefficient training.

2. **Activation Functions:** Many activation functions, like ReLU or sigmoid, work best when the input values are small. Large unscaled
   inputs could lead to saturation in some activation functions (e.g., sigmoid), where the gradient becomes very small, slowing down
   training.

3. **Weight Initialization:** Scaling data keeps the values close to zero, which aligns better with how neural network weights are often
   initialized. Without scaling, the model might not converge efficiently or could take much longer to train.

### Why Polyfit Doesn't Need Scaling:

Polynomial fitting (such as using `numpy.polyfit` or `scipy` tools) is a form of linear regression. Since polynomial fitting doesn't rely on
backpropagation and doesn't suffer from issues like vanishing gradients, it doesn't require scaling. The model simply fits the data using
least squares or a similar method, which is unaffected by the scale of the inputs.

### Skewing the Comparison:

The presence of scaling in the neural network fit **does** affect the comparison. If you directly compare the scaled neural network output
with the unscaled polynomial fit, the results would be biased. For a fair comparison, you'd need to **inverse-transform** the scaled neural
network output back to the original scale (as you're already doing with `output_scaler.inverse_transform`). This will allow you to compare
both fits on the same scale, making the comparison meaningful.

In short:

- **Scaling** helps the neural network perform well during training.
- **Polyfit** doesn’t need scaling since it's based on a different mathematical approach.
- Ensure both fits are on the same scale (preferably unscaled) before comparison to avoid skewed results.

In the case of comparing the two neural networks (one with 1 hidden layer and one with 4 hidden layers), it’s important to understand why
the additional layers might cause worse performance, especially when the data is mostly linear. Here's a breakdown of the reasoning:

### 1. **Overfitting on Simple Data:**

- **Simplicity of the Data:** If the data is mostly linear, a simple model (like the one with just 1 hidden layer) is often sufficient to
  capture the underlying pattern. Adding more layers increases the complexity of the model, which may lead to **overfitting**. Overfitting
  means the model becomes too specialized in learning the noise or insignificant variations in the training data, instead of capturing the
  general trend.
- In this case, the data may not require a deep neural network with multiple layers. A simpler model can perform better since it is not
  trying to fit unnecessary complexity.

### 2. **Vanishing/Exploding Gradients:**

- **Training Deep Networks:** Deep networks (with more layers) are prone to issues like **vanishing gradients**, where the gradients become
  too small to update the weights effectively. This results in poor learning in earlier layers. Alternatively, gradients might explode,
  leading to unstable updates.
- Even though you are using the **tanh** activation function, which is better than sigmoid in this regard, it can still suffer from
  vanishing gradients when multiple layers are involved.

### 3. **Higher Capacity Without a Need:**

- **Capacity of the Network:** A neural network with more layers and neurons has more capacity to learn complex patterns. However, when the
  underlying data is simple and linear, such capacity becomes unnecessary. The extra layers and parameters can lead to learning complex
  representations that do not generalize well to new data. This results in poorer performance during testing or validation.
- In your case, the MSE and MAE significantly increase with 4 layers, indicating the model may have started **over-complicating the fit**
  without gaining any benefit from the extra layers.

### 4. **Optimization Difficulty:**

- **Difficulty of Optimization:** Deeper networks can be harder to optimize because of the increased number of parameters and the challenge
  in propagating information across many layers. The optimizer might struggle with finding the optimal weights due to the larger search
  space introduced by the extra layers.
- You can see this reflected in your results, where the loss for the 4-layer model fluctuates more and never consistently improves compared
  to the 1-layer model. This suggests that the optimization is struggling with the deeper network, leading to higher losses.

### 5. **Bias-Variance Tradeoff:**

- **Bias-Variance Tradeoff:** Adding more layers reduces the bias of the model, making it more flexible in terms of fitting complex
  patterns. However, this also increases the **variance**, making the model more prone to overfitting, especially when the data doesn’t have
  complex underlying patterns.
- The 1-layer model strikes a better balance between bias and variance, as it doesn't need to reduce bias much in the case of mostly linear
  data, but adding more layers introduces too much variance.

### 6. **Slower Convergence:**

- **Slower Training:** More layers can cause slower convergence, and you can observe from the training loss that the deeper model struggles
  to improve over time. This could be due to the optimizer getting stuck in local minima or due to poor gradient flow.

### 7. **Instability with Dropout:**

- **Dropout Effect:** Although the dropout rate is relatively low (0.1), in deeper networks, dropout can make training unstable because
  random neurons are dropped during each iteration, and this effect is amplified across multiple layers. In shallower networks, the dropout
  effect is less pronounced and may lead to more stable training.

### Summary of the Two Models:

- **1 Hidden Layer (160 units):** Performs better because the network is simple and fits the mostly linear data appropriately. The model
  converges with good results (low MSE and MAE).
- **4 Hidden Layers (160 units each):** Adds unnecessary complexity, causing overfitting, optimization difficulties, and unstable training,
  resulting in significantly higher errors (MSE and MAE).

In conclusion, **more layers don’t always make the model better**. For simple, mostly linear data, a simpler model (with fewer layers) tends
to generalize better and trains more efficiently.
