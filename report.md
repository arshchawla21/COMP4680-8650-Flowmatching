# Part 1

### Data Visualisation

![image](res/part1/swiss_roll_visual.png)
![image](res/part1/gaussians_visual.png)
![image](res/part1/circles_visual.png)

### v-Prediction Flow Matching at `D = 2`

Hyperparameters:

| n_steps | lr | batch_size | denoising_steps | generated_points | clamping |
| - | - | - | - | - | - |
| 25000 | 1e-3 | 1024 | 50 | 2048 | 0.01 |

Trained on $v$-prediction (target) with $v$-loss

![image](res/part1/JiM_grid.png)

# Part 2

### Calculate flow matching based on $x$ or $v$-pred

We know (1) $z_t = (1-t)x + t\epsilon$. Further it is also defined that (2) $v = \epsilon - x$.

Rearranging (2) we get $\epsilon = v + x$, substituing this back into (1),

$$
z_t = (1-t)x + t(v + x) 
$$

$$
z_t = x - tx + tv + tx
$$

$$
z_t = x + tv \Rightarrow \boxed{x = z_t - tv} \Rightarrow \boxed{v = \frac{z_t - x}{t}}
$$

Hence in the scenario we predict $x$ with a loss in $v$, we can use the above equation to determine:

$$
v_\text{output} = \frac{z_t - x_\text{pred}}{t} \quad\quad\quad\quad v_\text{target} = \epsilon - x
$$

Similarly, for predicting $v$ with a loss in $x$, we have:

$$
x_\text{output} = z_t - t v_\text{pred} \quad\quad\quad\quad x_\text{target} = x
$$

### Reproduce papers findings

We use the same hyperparameters as Part 1.2.

![image](res/part2/swiss_roll_grid.png)
![image](res/part2/gaussians_grid.png)
![image](res/part2/circles_grid.png)

### Questions

#### 1. Prediction-type scaling

It is clear that $x$-prediction scales to $D=32$ better than $v$-prediction, across the three  datasets. Qualitatively $D=32$ with $x$-pred least resembles the target dataset. To make our analysis more robust, we measure generation density & coverage [1], a strong metric for comparing generation with ground truth. 

> **Density** is average number of real-point k-NN balls each generated sample falls into, divided by k [1]

> **Coverage** is fraction of real points whose k-NN ball contains at least one generated sample (k=5, ball radius = distance to k-th real neighbour). [1]

Below we plot density/coverage as a function of the ambient dimention $D$: 

![image](res/part2/density_coverage_vs_dimension.png)

The trend is clear, $v$-pred (green and red) scales the worse than $x$-pred (blue and orange). This is consistent with our generated results.

$$x_\text{pred} >_\text{dim scaling} v_\text{pred}$$

#### 2. Loss-space
The impact of loss space appears to depend on the prediction type. When inspecting the density/converage graph above, the $v$-pred appears to follow the exact same trend, with a slight offset. $x$-pred on the other hand displays similar scaling (over $D$) across the loss types, but the offset is signficant, with the blue (x-loss) being much lower than the orange (v-loss).

These observations are convincingly backed by the following normalised loss plot (where loss is scaled by starting value):

![image](res/part2/loss_curves_normalised.png)

The plots demonstrate that loss has minor impact on convergence for $v$-pred, while $v$-loss is significantly stronger for $x$-pred.

This being said, when observing the density/coverage figure, the dominating factor behind performance is certainly the prediction type, with loss having a minor impact in the $x$-pred scenario.

#### 3. Why does $x$-pred scale to higher $D$
$x$-prediction scales superior to $v$-prediction as $D$ grows, as the signal to be learned is better preserved. For $x$-pred at a high $D$, the target data lies on a 2D manifold under $R^D$ (well described in [Basics](papers/Basics.pdf)). Therefore the model is intrinsically learning in 2D, regardless of $D$, i.e., a projection onto the manifold. I.e., $rank(x)_\text{intrinsic} = 2$.

Meanwhile, for $v$-prediction, the model is tasked with learning $v=\epsilon-x$. Unlike $x$, $\epsilon$ is full-rank Guassian noise in $R^D$ (i.e., intrinsic dim = $D$). This implies $v$ measurments are dominated by $\epsilon$, meaning $\uparrow D$ results in linearly increased complexity in $v$, while $x$'s complexity remains at 2. From inductive bias, it follows that the model will struggle to fit the taret. 

As for the loss-space, the MSE is a $t$-dependent reweighting, meaning the underlying objective being learned remains the same.

# Part 3: Can We Rescue v-Prediction?
### Baseline
From part 2, we have the following baseline for the `swiss_roll` dataset:
![image](res/part3/swiss_roll_grid_baseline.png)

We previously determined that $v$-pred struggles as $D$ scales, due to full-rank gaussian noise impeding the intrisically 2-dimentional target signal. [2] outlines the key insight, **the process should lie on the same manifold as the data**. From this the we have our first proposition **reduce the intrinic dimenionality of noise to match data, independent of $D$**.

To achieve this, it is natural to use the randomly initalised matrix $P$, responsible for projecting $R^D \rightarrow R^2$. We outline the process as follows,

1. Sample $\epsilon_2 \sim \mathcal{N}(0,I)$, where $\epsilon_2 \in R^2$
2. Project to $R^D$: $\epsilon_D = \epsilon_2 \cdot P$, where $\epsilon_D \in R^D$ and $P \in R^{2 \times D}$

The same procedure must be applied at sampling time, since the model has only ever seen noise on the data subspace. Concretely, the initial state of the ODE is drawn as $z_1 = \epsilon_2 \cdot P$ rather than $z_1 \sim \mathcal{N}(0, I_D)$, sampling from full-rank Gaussian noise in $\mathbb{R}^D$ would push the trajectory off the data manifold and break the train/test consistency that this rescue depends on. Below are results using this noise dim optimisation:

![image](res/part3/swiss_roll_grid_opt001.png)

This had a profound positive impact on $v$-predictions.

We now turn our attention over to section 4 of [3], where it is outlined that **increasing the width of the network** (i.e., hidden layers) improves $v$-prediction. In light of this, we take our previous optimisation (noise dim), and test with `h=512` and `h=1024`.

![image](res/part3/swiss_roll_grid_opt002_h512.png)

![image](res/part3/swiss_roll_grid_opt002_h1024.png)

While the impact is not quite as notesable, resultant generations appear to better follow the ground truth shape, although there is not much to split `h=512` and `h=1024` from a qualitative standpoint.

Raw data (the graphs are based on)

| stage | h | D | pred | loss | params | size | time | final loss |
|---|---|---|---|---|---|---|---|---|
| baseline | 256 | 2 | x | x | 297218 | 1.19 MB | 0m 47s | 0.2632 |
| baseline | 256 | 2 | x | v | 297218 | 1.19 MB | 0m 45s | 0.9808 |
| baseline | 256 | 2 | v | x | 297218 | 1.19 MB | 0m 48s | 0.2637 |
| baseline | 256 | 2 | v | v | 297218 | 1.19 MB | 0m 46s | 0.9451 |
| baseline | 256 | 8 | x | x | 300296 | 1.21 MB | 0m 49s | 0.0658 |
| baseline | 256 | 8 | x | v | 300296 | 1.21 MB | 0m 45s | 0.2475 |
| baseline | 256 | 8 | v | x | 300296 | 1.21 MB | 0m 47s | 0.0666 |
| baseline | 256 | 8 | v | v | 300296 | 1.21 MB | 0m 46s | 0.2385 |
| baseline | 256 | 32 | x | x | 312608 | 1.26 MB | 0m 48s | 0.0165 |
| baseline | 256 | 32 | x | v | 312608 | 1.26 MB | 0m 45s | 0.0607 |
| baseline | 256 | 32 | v | x | 312608 | 1.26 MB | 0m 49s | 0.0185 |
| baseline | 256 | 32 | v | v | 312608 | 1.26 MB | 0m 46s | 0.0716 |
| opt001 | 256 | 2 | x | x | 297218 | 1.19 MB | 0m 45s | 0.2629 |
| opt001 | 256 | 2 | x | v | 297218 | 1.19 MB | 0m 47s | 0.9768 |
| opt001 | 256 | 2 | v | x | 297218 | 1.19 MB | 0m 47s | 0.2637 |
| opt001 | 256 | 2 | v | v | 297218 | 1.19 MB | 0m 46s | 0.9451 |
| opt001 | 256 | 8 | x | x | 300296 | 1.21 MB | 0m 46s | 0.0658 |
| opt001 | 256 | 8 | x | v | 300296 | 1.21 MB | 0m 47s | 0.2438 |
| opt001 | 256 | 8 | v | x | 300296 | 1.21 MB | 0m 47s | 0.0659 |
| opt001 | 256 | 8 | v | v | 300296 | 1.21 MB | 0m 46s | 0.2357 |
| opt001 | 256 | 32 | x | x | 312608 | 1.26 MB | 0m 47s | 0.0165 |
| opt001 | 256 | 32 | x | v | 312608 | 1.26 MB | 0m 49s | 0.0607 |
| opt001 | 256 | 32 | v | x | 312608 | 1.26 MB | 0m 48s | 0.0165 |
| opt001 | 256 | 32 | v | v | 312608 | 1.26 MB | 0m 47s | 0.0587 |
| opt002_h512 | 512 | 2 | x | x | 1118722 | 4.48 MB | 0m 47s | 0.2631 |
| opt002_h512 | 512 | 2 | x | v | 1118722 | 4.48 MB | 0m 47s | 0.9883 |
| opt002_h512 | 512 | 2 | v | x | 1118722 | 4.48 MB | 0m 47s | 0.2633 |
| opt002_h512 | 512 | 2 | v | v | 1118722 | 4.48 MB | 0m 46s | 0.9444 |
| opt002_h512 | 512 | 8 | x | x | 1124872 | 4.50 MB | 0m 46s | 0.0657 |
| opt002_h512 | 512 | 8 | x | v | 1124872 | 4.50 MB | 0m 47s | 0.2419 |
| opt002_h512 | 512 | 8 | v | x | 1124872 | 4.50 MB | 0m 47s | 0.0659 |
| opt002_h512 | 512 | 8 | v | v | 1124872 | 4.50 MB | 0m 46s | 0.2349 |
| opt002_h512 | 512 | 32 | x | x | 1149472 | 4.60 MB | 0m 48s | 0.0165 |
| opt002_h512 | 512 | 32 | x | v | 1149472 | 4.60 MB | 0m 49s | 0.0605 |
| opt002_h512 | 512 | 32 | v | x | 1149472 | 4.60 MB | 0m 49s | 0.0165 |
| opt002_h512 | 512 | 32 | v | v | 1149472 | 4.60 MB | 0m 48s | 0.0589 |
| opt002_h1024 | 1024 | 2 | x | x | 4334594 | 17.34 MB | 1m 13s | 0.2635 |
| opt002_h1024 | 1024 | 2 | x | v | 4334594 | 17.34 MB | 1m 13s | 0.9721 |
| opt002_h1024 | 1024 | 2 | v | x | 4334594 | 17.34 MB | 1m 14s | 0.2636 |
| opt002_h1024 | 1024 | 2 | v | v | 4334594 | 17.34 MB | 1m 13s | 0.9446 |
| opt002_h1024 | 1024 | 8 | x | x | 4346888 | 17.39 MB | 1m 13s | 0.0658 |
| opt002_h1024 | 1024 | 8 | x | v | 4346888 | 17.39 MB | 1m 13s | 0.2444 |
| opt002_h1024 | 1024 | 8 | v | x | 4346888 | 17.39 MB | 1m 13s | 0.0659 |
| opt002_h1024 | 1024 | 8 | v | v | 4346888 | 17.39 MB | 1m 13s | 0.2357 |
| opt002_h1024 | 1024 | 32 | x | x | 4396064 | 17.59 MB | 1m 15s | 0.0165 |
| opt002_h1024 | 1024 | 32 | x | v | 4396064 | 17.59 MB | 1m 15s | 0.0612 |
| opt002_h1024 | 1024 | 32 | v | x | 4396064 | 17.59 MB | 1m 15s | 0.0165 |
| opt002_h1024 | 1024 | 32 | v | v | 4396064 | 17.59 MB | 1m 15s | 0.0588 |

for 1) the stuff from the papers (two references) speaks on if failure is fundemental

for 2) list approaches, compare compute with the x-pred default 

for 3) compare x-pred and v-pred respond

![image](res/part3/scaling_dens_cov_per_predloss.png)

from graph x pred stays reasonally constant, but v pred big improvement particually baseline->opt

for 4)

- seperate ground truth to left
- add loss curves to part 2) and part 3)
- add increased hidden layer to part 3)

- final part steps y, 

# References
[1] Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", ICML 2020 (arXiv:2002.09797).

[2] Li et al. "Back to Basics: Let Denoising Generative Models Denoise", 2026 (arXiv:2511.13720v2)

[3] Zheng et al. "Diffusion Transformers with Representational Autoencoders", 2025 (arXiv:2510.11690v1)