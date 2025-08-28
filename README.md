# PINNS

kalman_face_tracking.ipynb: a minimal kalman example for face tracking

pinn_solution.ipynb: minimal example of a PINN for the solution of a simple pde

pinn_discovery.ipynb: minimal example of a PINN for the discovery of a simple pde

---

**PINNs** solve any supervised learning task with a dynamical model constraint on the data.

Two subproblems that PINNs face are

1. **Data-driven solutions** (solving): Solving the dynamics.

2. **Data-driven discovery** (model identification): modeling unknwon non-linearity of given dynamics with a NN.

## PINN for solving PDEs

PINNs formulate the PDE solution problem by including initial and boundary conditions into the loss function of a neural network as soft constraints.

Let
$$
\dfrac{dx}{dt} = f(x,t), t \in [0,T]
$$
be known dynamics of $x(t)$ and $\{x_i,t_i\}_{i=0}^{N-1}$ be given observations with $i=0$ the initial conditions.

e.g.
$$
\dfrac{dx}{dt} = t^2\\
x_0 = 0
$$
Wanted (ground truth) solution:

$$ x(t) = \dfrac{1}{3} t^3 $$

### Classical Solver (e.g. Euler)

Starting from initial conditions $x_0$ it propagates predictions forward using some order of approximation around $x_{n-1}$:

$$ x(t_n) = x(t_{n-1}) + h f(x(t_{n-1}), t_{n-1}) $$

### PINN solver

A neural net $g_{\theta}$ approximates $x(t)$ given $(x_i,t_i)$:

$$ g(t_i; \theta) \rightarrow x_{i+1} $$

which we train with 
$$ L_1 = \sum_i ||g(t_i) - x_{i}||_2^2 $$
$$ L_2 = \sum_i ||\dfrac{dg}{dt}(t_i) - f_i||_2^2 $$

Let $g(t)$ be a neural approximates $f$ and is trained using ground thuth measurements of $u$ and $v$ and the p.d.e as a contraint

$$ \min L = \min  L_1 + L_2 \quad s.t. \quad f(u(x,t),v(x,t),...) = 0 $$

## Study material

[IDRL-Lab](https://github.com/idrl-lab/PINNpapers.git) study material collection on PINNs

[iPINNs: incremental learning for Physics‑informed neural networks](https://epubs.siam.org/doi/abs/10.1137/20m1318043?journalCode=sjoce3)

    "..finding a set of neural network parameters that fulfill a PDE at the boundary and within the domain of interest can be challenging and non-unique due to the complexity of the loss landscape that needs to be traversed.".

[Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks](https://epubs.siam.org/doi/abs/10.1137/20m1318043?journalCode=sjoce3)

    Exploding gradients(?) "..numerical stiffness leading to unbalanced back-propagated gradients during model training." 

### Code

[Vanilla PINN](https://github.com/rezaakb/pinns-torch)

[Physics Informed Transformers](https://github.com/AdityaLab/pinnsformer)


PINNS solve continuous or discrete dynamics
Continuous model: data-efficient spatio-temporal function approximation
Discrete model: Runge-Kutta time stepping schemes of arbitrary accuracy and unlimited number of stages. -->
