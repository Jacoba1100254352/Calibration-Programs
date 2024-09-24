What you are describing is a **calibration process** where the goal is to establish a reliable relationship between the **Instron force
measurements (N)**, which serve as your baseline or reference data, and the **sensor ADC values**, which are being measured by your Arduino.
This calibration ensures that the sensor’s output (in ADC values) can be accurately mapped to the real-world Instron force values, providing
a meaningful interpretation of sensor data.

### Polyfit Calibration Process (Current Approach):

In the polyfit approach, the method is performing **polynomial regression**:

1. **Instron Force (N)** is the baseline (the independent variable).
2. **Sensor ADC values** are the output you want to map to the Instron force (the dependent variable).
3. **Polynomial Fitting**: By fitting polynomials of different orders (linear, quadratic, cubic, etc.) to the data, you're checking how well
   different polynomial models can represent the relationship between the Instron force and the sensor ADC values.
    - **First-order (linear fit)**: Assumes a simple linear relationship between Instron force and ADC values.
    - **Second-order, third-order fits**: These allow more complex, curved relationships between the two variables.
4. **Residuals**: After fitting the polynomial, you calculate the residuals, which represent the difference between the actual ADC values
   and the values predicted by the polynomial. A good fit would have small residuals, meaning the polynomial closely approximates the real
   relationship.

This approach is effectively creating a **calibration function** where:

- Given an ADC value from the sensor, you can predict the corresponding Instron force (N) based on the polynomial equation.
- The **goal is to find a function** that best maps the sensor's raw output (ADC values) to the baseline (Instron force).

### Neural Fit Approach (Your Goal):

In your case, you want to replace or compare the polynomial fitting process with a **neural network** fit (a form of machine learning
model). Here's how the neural fit process is supposed to work in parallel to polyfit:

1. **Instron Force (N)** is still the baseline (independent variable).
2. **Sensor ADC values** are the output that you want the neural network to learn to predict (dependent variable).
3. **Neural Network Fit**: Instead of using polynomial equations, you are using a neural network to learn the relationship between the
   Instron force and the sensor ADC values.
    - The **training process** involves feeding the network pairs of Instron force values (inputs) and corresponding sensor ADC values (
      targets/outputs) so that the network can learn the relationship.
    - The network learns by adjusting its internal parameters (weights and biases) to minimize the difference between its predicted ADC
      values and the actual sensor ADC values (similar to how the polyfit minimizes residuals).

### Key Questions:

#### 1. **Is the neural fit process working in a similar way to polyfit?**

Yes, in concept, the **neural fit** process is intended to do the same thing as polyfit but using a more flexible, data-driven approach.
Just like polynomial fitting, the neural network is trying to **learn the relationship** between the Instron force (N) and the ADC values,
so that you can later use the network to predict the ADC values given the Instron force.

#### 2. **Is the "training" on the same data it’s supposed to be calibrating/correlating correct?**

- **Yes**, you are training the neural network on **the same data** that you want to calibrate. The training process involves the network
  learning to map Instron force to ADC values based on the data it is given.
- The difference between polyfit and neural fit is that **polyfit uses a fixed mathematical model** (a polynomial equation), while the
  neural network tries to **learn a more flexible, non-linear model** from the data itself.

Since your data includes **Instron Force (input)** and **ADC values (output)**, it is correctly structured for training. The network is
supposed to learn the mapping from the Instron baseline (N) to the sensor values (ADC), similar to what polyfit is doing, but with a more
adaptive approach.

#### 3. **Is it correct to train the model on the same data it's supposed to be calibrating?**

**Yes**, this is typical in calibration tasks. You are using the data you already have (Instron N and sensor ADC values) to create a
calibration model that can predict one from the other.

- In supervised learning, it is standard practice to train the model using the same data that you are calibrating. After training, the model
  should generalize well enough to predict ADC values for any new Instron force values (or vice versa, depending on the direction of your
  calibration).

In your case, the goal is for the neural network to establish an accurate mapping from **Instron N** to **sensor ADC values** (or the
reverse). Once trained, the model should ideally be able to predict the ADC values for any given Instron force.

### Comparison Between Polyfit and Neural Fit:

- **Polyfit**: Uses fixed, predefined functions (polynomials) to describe the relationship. The degree of the polynomial controls the
  complexity of the relationship.
- **Neural Fit**: Uses a flexible model (neural network) that can approximate more complex, non-linear relationships. The number of layers
  and neurons controls the network’s complexity.

#### When Neural Fit May Be Beneficial:

- **Non-linearity**: If the relationship between the Instron force and the sensor ADC values is **non-linear** and cannot be easily captured
  by a polynomial, a neural network can often find a better fit since it doesn’t rely on a specific mathematical form.
- **More Flexibility**: Neural networks can capture relationships that polynomials might miss, especially if the data contains complex
  patterns or interactions.

### Conclusion:

- The neural fit process is indeed supposed to **calibrate** the relationship between **Instron force (N)** and **sensor ADC values**,
  similar to how the polyfit process works.
- The data you are providing for training (Instron N as input, ADC as output) is correct for this task.
- The "training" process is appropriate because you are teaching the model to learn the relationship from the baseline (Instron) to the
  output (ADC), which is exactly what you are trying to calibrate.
