# energy-based-generative-mnist

# Deep Energy-Based Models on MNIST  
**DGM Homework 3 – Question 1**

This repository contains the full implementation for **Question 1 (Energy-Based Models)** of the *Deep Generative Models* course assignment.  
The project focuses on training a **Deep Energy-Based Model (EBM)** on the MNIST dataset using **Langevin Dynamics**, with an emphasis on **training stability, sampling quality, and denoising behavior**.

The accompanying written report, submitted separately under the author’s name (*Vali*), provides the theoretical derivations, experimental analysis, and qualitative results. This repository contains **only the code corresponding to Question 1**.

---

## Project Overview

Energy-Based Models define an unnormalized probability density of the form:

\[
p_\theta(x) \propto \exp(-E_\theta(x))
\]

where a neural network learns an energy function that assigns low energy to real data and higher energy elsewhere.  
Since the normalization constant is intractable, training relies on **contrastive methods** and **MCMC-based sampling**, rather than explicit likelihood computation.

In this project, we implement a **deep convolutional EBM** and train it on MNIST using **Langevin Dynamics** to approximate negative samples.

---

## Key Contributions

- End-to-end implementation of a **Deep Energy-Based Model** for image modeling  
- **Langevin Dynamics sampling** with gradient-based updates and noise injection  
- **Replay Buffer (Persistent Contrastive Divergence)** to improve sampling efficiency  
- **Adaptive stabilization mechanisms** to prevent divergence during training  
- Qualitative evaluation via image generation, comparison with real samples, and denoising experiments

---


## Training Methodology

### Model Architecture
- Convolutional neural network producing a **scalar energy**
- Spectral normalization to stabilize gradients
- Smooth nonlinear activations for a well-behaved energy landscape

### Sampling
Negative samples are generated using **Langevin Dynamics**:

\[
x_{t+1} = x_t - \frac{\epsilon^2}{2} \nabla_x E_\theta(x_t) + \epsilon z_t
\]

with gradient clipping, pixel clamping to \([0,1]\), and optional initialization from a replay buffer.

### Stability Techniques
Training EBMs is highly unstable. To mitigate this, the implementation includes:

- Replay buffer sampling (Persistent Contrastive Divergence)
- Adaptive learning rate decay on instability
- Dynamic regularization adjustment
- Checkpoint rollback on divergence
- Gradient norm clipping

These mechanisms are critical for preventing energy collapse and runaway gradients.

---

## Evaluation (Qualitative)

No numerical likelihood-based metric is used in this question.  
Evaluation is performed visually in the notebook through:

1. Image generation from uniform noise using Langevin sampling  
2. Side-by-side comparison with real MNIST samples  
3. Denoising experiments at multiple noise levels  

These evaluations directly probe whether the learned energy landscape captures the MNIST data manifold.
