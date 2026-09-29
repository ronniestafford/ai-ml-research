<!-- \providecommand{\mat}[1]{\mathbf{#1}}
\providecommand{\vctr}[1]{\mathbf{#1}}
\providecommand{\spaces}[1]{\mathcal{#1}}
\providecommand{\numberset}[1]{\mathbb{#1}}
\providecommand{\sml}[1]{{#1}}
\providecommand{\ele}[1]{\mathrm{#1}}
\providecommand{\mapdefs}{{{:}}}
\providecommand{\from}{{\leftarrow}}
\providecommand{\tos}{{\to}}
\providecommand{\ins}{\in}
\providecommand{\sims}{{\sim}}
\providecommand{\timess}{{\times}}
\providecommand{\imp}{\Rightarrow}
\providecommand{\twospace}{\space\space}
\providecommand{\argmin}{\mathrm{argmin}}
\providecommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\providecommand{\argminu}[2]{{\underset{#1}\argmin{[#2]}}}
\providecommand{\bigargminu}[2]{{\underset{#1}\argmin{\big[#2\big]}}}
\providecommand{\biggargminu}[2]{{\underset{#1}\argmin{\bigg[#2\bigg]}}} -->

# Backprop

## Risk Minimisation

Assume a joint distribution $P_{X,Y}$ over the input space $𝓧$, output space $𝒴$, function space $ℱ$ and loss function $𝓛$, population risk $R$, then the general form of the minimisation objective is,

$$
\begin{aligned}
f^{*}&=\underset{f\in ℱ}{\mathrm{argmin}}\left[R(f)\right],\\
&=\underset{f\in ℱ}{\mathrm{argmin}} \bigg[ 𝔼_{P_{X,Y}}\left[ 𝓛(f,X,Y) \right] \bigg].
\end{aligned}
$$

The solution $f$ and what it represents depends on i) the form of $P_{X,Y}$ (e.g. its factorisation), ii) function space $ℱ$ and iii) choice of loss function $𝓛$.

## Supervised Learning

If the joint distribution is factorised $P_{X,Y}= P_{X}P_{Y|X}$,
$f{{:}}𝓧{\to}𝒴$ and loss function is chosen as $𝓛{{:}}𝒴{\times}𝒴{\to}ℝ$, then the risk minimisation problem becomes

$$
\begin{aligned}
f^{*}&=\underset{f\in ℱ}{\mathrm{argmin}}{ \bigg[𝔼_{P_{X}} 𝔼_{P_{Y|X}}\left[  𝓛(f(X),Y) \right] \bigg] }.
\end{aligned}
$$

This is the standard supervised learning problem where $P_{X}$ is the data generator distribution and $f$ is modelling some aspect of the conditional distribution. The aspect of $P_{Y|X}$ that is modelled is determined by the choice of loss function e.g. squared loss determines that $f^{*}$ models the conditional mean $\mu_{Y|X=x} \coloneqq 𝔼_{P_{Y|X=x}}[Y]$ under the conditional distribution, similarly cross-entropy targets the conditional distribution $P(Y|X= x)$ itself etc.

Empirical risk $\hat{R}$ minimisation requires a finite sample data set $\lbrace(x_{d},y_{d})\rbrace_{d=1}^{D} \overset{\mathrm{i.i.d.}}{\sim} P_{X,Y}$, such that

$$
\begin{aligned}
f^{*}&=\underset{f\in ℱ}{\mathrm{argmin}}{ \bigg[ 𝔼_{P_{Y|X}} \left[  𝓛(f(X),Y) \right] \bigg] },\\
&=\underset{f\in ℱ}{\mathrm{argmin}}{\bigg[\frac{1}{D}\sum_{d= 1}^{D}𝓛(f(x_{d}),y_{d})\bigg]}.
\end{aligned}
$$

## Chain Rule

A deep neural network $f_{\Theta}{{:}}𝓧{\to}𝒴$ is a composite function of $L$ layers, where each $l^{th}$ layer is itself a composition of a non-linear function $h^{(l)}(\cdot)$ and affine transformation $g_{\theta^{(l)}}^{l}(\cdot)$. The set of parameters for the entire deep network is $\Theta\coloneqq(\theta^{(L)}, \cdots, \theta^{(l)}, \cdots \theta^{(1)})$, where $\theta^{(l)}$ parameterises $g^{(l)}$,

$$
\begin{aligned}
f_{\Theta}(x)\coloneqq&(h^{(L)} \circ g^{(L)} \circ\cdots\circ h^{(l)} \circ g^{(l)} \circ\cdots\circ h^{(1)} \circ g^{(1)})(x),
\end{aligned}
$$

Then by the chain rule

$$
\begin{aligned}
\frac{\partial f_{\Theta}(x)}{\partial \theta^{(l)}}=&\left(\frac{\partial h^{(L)}}{\partial g^{(L)}}\frac{\partial g^{(L)}}{\partial h^{(L-1)}}  \cdots \frac{\partial h^{(l)}}{\partial g^{(l)}}\frac{\partial g^{(l)}}{\partial \theta^{(l)}}\right)(x),
\end{aligned}
$$

where each partial derivative is evaluated after $𝐱$ is propagated in a forward pass through the network.

## Gradient Descent

The gradient update rule for layer $l$ at a particular $x_{d}\in 𝓧$ is

$$
\begin{aligned}
\theta^{(l)} &{\leftarrow} \theta^{(l)}-\frac{\eta}{D}\sum_{d= 1}^{D}\left(\frac{\partial 𝓛_{d}}{\partial f_{\Theta}}\frac{\partial f_{\Theta}}{\partial \theta^{(l)}}\right)(x_{d}),
\end{aligned}
$$

where $\eta>0$ is the learning rate. The process of making a forward pass of $x_{d}$ through the entire network, calculating the loss function $𝓛_{d}$ with the target $y_{d} \in 𝒴$, then calculating the partial derivative tensors in a backwards pass is known as **backpropagation**. Usually each parameter update is done with a gradient calculated for a minibatch $M{\ll} D$.

## Concrete Example

### Matrix \& Vector Notation

Let vectors $𝐱\in 𝓧$, $\mathbf{y}\in 𝒴$, then the neural network $f_{\Theta}$ consist of the affine transformation

$$
\begin{aligned}
𝐠^{(l)}&=g(𝐡^{(l-1)}),\\
&=𝐖^{(l)}𝐡^{(l-1)}+𝐛,
\end{aligned}
$$

and non-linearity (piecewise-linear)

$$
\begin{aligned}
𝐡^{(l)}&= h(𝐠^{(l)}),\\
&=\mathrm{ReLU}(𝐠^{(l)}),\\
&=\max(0,𝐠^{(l)}),
\end{aligned}
$$

where the ReLU acts element-wise $\forall \text{g} \in 𝐠$,

$$
\begin{aligned}
\mathrm{max}(0,\text{g})&=\begin{cases}
0, & \text{g} \leq 0\\
\text{g}, & \text{g} > 0
\end{cases},
\end{aligned}
$$

with $𝐡^{(0)}=𝐱\in 𝓧$, $𝐡^{(L)}=\mathbf{y}\in 𝒴$, $𝐖^{(l)}\in ℝ^{m{\times} n}$, such that $g{{:}}ℝ^{n}{\to}ℝ^{m}$, $h{{:}}ℝ^{m}{\to}ℝ^{m}$. Note that $𝐠^{(l)}$ and $𝐡^{(l)}$ are calculated by propagating $𝐱$ forward through the network. Strictly each layer output dimension $m$ is arbitrarily chosen and $n$ equals the previous layer's $m$.

### Partial Derivative Tensors

The chain rule is now

$$
\begin{aligned}
\frac{\partial 𝓛_{d}}{\partial 𝐖^{(l)}}=&\left(\frac{\partial 𝓛_{d}}{\partial h^{(L)}}\frac{\partial h^{(L)}}{\partial g^{(L)}}\frac{\partial g^{(L)}}{\partial h^{(L-1)}}  \cdots \frac{\partial h^{(l)}}{\partial g^{(l)}}\frac{\partial g^{(l)}}{\partial 𝐖^{(l)}}\right)(𝐱_{d}),
\end{aligned}
$$

where from the previous notation,

$$
\Theta\coloneqq(𝐖^{(L)}, \cdots, 𝐖^{(l)}, \cdots 𝐖^{(1)}).
$$

Temporarily dropping the layer index and introducing tensor indexes $ijk$, vector elements $\text{g}_k \in 𝐠$, $\text{h}_j \in 𝐡$, $\text{w} \in 𝐖$ and $\text{b}_k \in 𝐛$, then the affine transformation is

$$
\begin{aligned}
\text{g}_{k} &= \sum_{j}\text{w}_{kj}\text{h}_{j} + b_{k}.
\end{aligned}
$$

This leads to the partial derivative w.r.t weights

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial w_{i,j}} &= \text{h}_{j}\delta_{ik},\\
\Rightarrow \frac{\partial \text{g}_{k}}{\partial 𝐖} &= \mathbf{e}_{k}𝐡^{\top}\space\space\in ℝ^{m\times n},
\end{aligned}
$$

where the standard $k^{th}$ basis vector is $\mathbf{e}_{k}$, the partial derivative w.r.t layer inputs

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial h_{j}} &= \text{w}_{kj},\\
\Rightarrow \frac{\partial 𝐠}{\partial 𝐡} &= 𝐖\space\space\in ℝ^{m\times n},
\end{aligned}
$$

partial derivative w.r.t intercept

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial b_{i}} &= \delta_{ki},\\
\Rightarrow \frac{\partial 𝐠}{\partial 𝐛} &= \mathbf{I}_{m}\space\space\in ℝ^{m\times m},
\end{aligned}
$$

and partial derivative of the piecewise linear function is,

$$
\begin{aligned}
\frac{\partial \text{h}_{j}}{\partial g_{k}} &= \delta_{jk}1_{\text{g}_{k}>0},\\
\Rightarrow \frac{\partial 𝐡}{\partial 𝐠} &= \mathrm{diag}(\mathbf{1}_{\text{g}>0})\space\space\in ℝ^{m\times m}.
\end{aligned}
$$

If the loss function is a squared loss and the output of the neural network after a forward pass is $f_{\Theta}(𝐱_{d})=𝐡^{(L)}=𝐟_{d}$, then the empirical risk minimisation objective for a sample $(𝐱_{d},\mathbf{y}_d)$ is,

$$
\begin{aligned}
𝓛_{d}&=\frac{1}{2}{||f_{\Theta}(𝐱_{d})-\mathbf{y}_{d}||}^{2}_{2},\\
&=
\frac{1}{2}{(𝐟_{d}-\mathbf{y}_{d})}^{\top}(𝐟_{d}-\mathbf{y}_{d}),\\
&= \frac{1}{2}𝐟_{d}^{\top}𝐟_{d}-2\mathbf{y}_{d}^{\top}𝐟_{d}+\mathbf{y}_{d}^{\top}\mathbf{y}_{d},\\
\Rightarrow \frac{\partial 𝓛_{d}}{\partial 𝐟_{d}}&=𝐟_{d}^{\top}-\mathbf{y}_{d}^{\top}\space\space\in ℝ^{1\times\mathrm{dim}({𝒴})}.
\end{aligned}
$$

### Putting it all Together

For each sample $(𝐱_{d},\mathbf{y}_d)$, the network and loss function are evaluated in a forward pass. All of the partial derivative tensors are then calculated and multiplied together to get the weight updates. Each weight update is evaluated for a minibatch in stochastic gradient descent. All of the tensors are two-dimension, apart from the three-dimensional weight partial derivative, $\frac{\partial 𝐠}{\partial 𝐖}$. In this case, the matrix-tensor multiplication operates along the tensor's third dimension (k-index),

$$
\begin{aligned}
\frac{\partial 𝓛}{\partial 𝐖^{(l)}}&=\frac{\partial 𝓛}{\partial 𝐠^{(l)}}\frac{\partial 𝐠^{(l)}}{\partial 𝐖^{(l)}},\\
&=\sum_{k}\frac{\partial 𝓛}{\partial \text{g}_{k}^{(l)}}\frac{\partial \text{g}_{k}^{(l)}}{\partial 𝐖^{(l)}}\space\space\in ℝ^{m{\times} n}.
\end{aligned}
$$
