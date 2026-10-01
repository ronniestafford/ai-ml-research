# Backprop

## Risk Minimisation

Assume a joint distribution $P_{X,Y}$ over the input space $\mathcal{X}$, output space $\mathcal{Y}$, function space $\mathcal{F}$ and loss function $\mathcal{L}$, population risk $R$, then the population risk minimisation objective is,

$$
\begin{aligned}
f^{*}&=\underset{f\in \mathcal{F}}{\mathrm{argmin}}\left[R(f)\right],\\
&=\underset{f\in \mathcal{F}}{\mathrm{argmin}} \bigg[ \mathbb{E}_{P_{X,Y}}\left[ \mathcal{L}(f,X,Y) \right] \bigg].
\end{aligned}
$$

The solution $f$ and what it represents depends on i) the form of $P_{X,Y}$ (e.g. its factorisation), ii) function space $\mathcal{F}$ and iii) choice of loss function $\mathcal{L}$.

## Supervised Learning

If the joint distribution is factorised $P_{X,Y}=P_{X}P_{Y|X}$,
$f\colon\mathcal{X}{\to}\mathcal{Y}$ and loss function is chosen as $\mathcal{L}\colon\mathcal{Y}{\times}\mathcal{Y}{\to}\mathbb{R}$, then the risk minimisation problem becomes

$$
\begin{aligned}
f^{*}&=\underset{f\in \mathcal{F}}{\mathrm{argmin}}{ \bigg[\mathbb{E}_{P_{X}} \mathbb{E}_{P_{Y|X}}\left[  \mathcal{L}(f(X),Y) \right] \bigg] }.
\end{aligned}
$$

This is the standard supervised learning problem where $P_{X}$ is the data generator distribution and $f$ is modelling some aspect of the conditional distribution. The aspect of $P_{Y|X}$ that is modelled is determined by the choice of loss function e.g. squared loss determines that $f^{*}$ models the conditional mean $\mu_{Y|X=x} \coloneqq \mathbb{E}_{Y|X=x}[Y]$, similarly a cross-entropy loss targets the conditional distribution $P(Y|X=x)$ itself etc.

Empirical risk $\hat{R}$ minimisation proceeds by drawaing a finite sample data set $\lbrace(x_{d},y_{d})\rbrace_{d=1}^{D} \overset{\mathrm{i.i.d.}}{\sim} P_{X,Y}$, such that

$$
\begin{aligned}
f^{*}&=\underset{f\in \mathcal{F}}{\mathrm{argmin}}{\bigg[\frac{1}{D}\sum_{d=1}^{D}\mathcal{L}(f(x_{d}),y_{d})\bigg]}.
\end{aligned}
$$

## Chain Rule

A deep neural network $f_{\Theta}\colon\mathcal{X}{\to}\mathcal{Y}$ is a composite function of $L$ layers, where each $l^{th}$ layer is itself a composition of a non-linear function $h^{(l)}(\cdot)$ and affine transformation $g_{\theta^{(l)}}^{l}(\cdot)$. The set of parameters for the entire deep network is $\Theta\coloneqq(\theta^{(L)}, \cdots, \theta^{(l)}, \cdots \theta^{(1)})$, where $\theta^{(l)}$ parameterises $g^{(l)}$,

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

where each partial derivative is evaluated after $\mathbf{x}$ is propagated in a forward pass through the network.

## Gradient Descent

The gradient update rule for layer $l$ at a particular $x_{d}\in \mathcal{X}$ is

$$
\begin{aligned}
\theta^{(l)} &{\leftarrow} \theta^{(l)}-\frac{\eta}{D}\sum_{d=1}^{D}\left(\frac{\partial \mathcal{L}_{d}}{\partial f_{\Theta}}\frac{\partial f_{\Theta}}{\partial \theta^{(l)}}\right)(x_{d}),
\end{aligned}
$$

where $\eta>0$ is the learning rate. The process of making a forward pass of $x_{d}$ through the entire network, calculating the loss function $\mathcal{L}_d$ with the target $y_d\in\mathcal{Y}$, then calculating the partial derivative tensors in a backwards pass is known as **backpropagation**. Usually each parameter update is made with a gradient calculated for a minibatch size $M{\ll} D$ sampled from the training set.

## Concrete Example

### Matrix \& Vector Notation

Let vectors $\mathbf{x}\in \mathcal{X}$, $\mathbf{y}\in \mathcal{Y}$, then the neural network $f_{\Theta}$ consist of the affine transformation

$$
\begin{aligned}
\mathbf{g}^{(l)}&=g(\mathbf{h}^{(l-1)}),\\
&=\mathbf{W}^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b},
\end{aligned}
$$

and non-linearity (piecewise-linear)

$$
\begin{aligned}
\mathbf{h}^{(l)}&=h(\mathbf{g}^{(l)}),\\
&=\mathrm{ReLU}(\mathbf{g}^{(l)}),\\
&=\max(0,\mathbf{g}^{(l)}),
\end{aligned}
$$

where the ReLU acts element-wise $\forall \text{g} \in \mathbf{g}$,

$$
\begin{aligned}
\mathrm{max}(0,\text{g})&=\begin{cases}
0, & \text{g} \leq 0\\
\text{g}, & \text{g} > 0
\end{cases},
\end{aligned}
$$

with $\mathbf{h}^{(0)}=\mathbf{x}\in \mathcal{X}$, $\mathbf{h}^{(L)}=\mathbf{y}\in \mathcal{Y}$, $\mathbf{W}^{(l)}\in \mathbb{R}^{m{\times} n}$, such that $g\colon\mathbb{R}^{n}{\to}\mathbb{R}^{m}$, $h\colon\mathbb{R}^{m}{\to}\mathbb{R}^{m}$. Note that $\mathbf{g}^{(l)}$ and $\mathbf{h}^{(l)}$ are calculated by propagating $\mathbf{x}$ forward through the network. Strictly each layer output dimension $m$ is arbitrarily chosen and $n$ equals the previous layer's $m$.

### Partial Derivative Tensors

The chain rule is now

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{d}}{\partial \mathbf{W}^{(l)}}=&\left(\frac{\partial \mathcal{L}_{d}}{\partial h^{(L)}}\frac{\partial h^{(L)}}{\partial g^{(L)}}\frac{\partial g^{(L)}}{\partial h^{(L-1)}}  \cdots \frac{\partial h^{(l)}}{\partial g^{(l)}}\frac{\partial g^{(l)}}{\partial \mathbf{W}^{(l)}}\right)(\mathbf{x}_{d}),
\end{aligned}
$$

where from the previous notation,

$$
\Theta\coloneqq(\mathbf{W}^{(L)}, \cdots, \mathbf{W}^{(l)}, \cdots \mathbf{W}^{(1)}).
$$

Temporarily dropping the layer index and introducing tensor indexes $ijk$, vector elements $\text{g}_k \in \mathbf{g}$, $\text{h}_j \in \mathbf{h}$, $\text{w} \in \mathbf{W}$ and $\text{b}_k \in \mathbf{b}$, then the affine transformation is

$$
\begin{aligned}
\text{g}_{k} &=\sum_{j}\text{w}_{kj}\text{h}_{j} + b_{k}.
\end{aligned}
$$

This leads to the partial derivative w.r.t weights

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial w_{i,j}} &=\text{h}_{j}\delta_{ik},\\
\Rightarrow \frac{\partial \text{g}_{k}}{\partial \mathbf{W}} &=\mathbf{e}_{k}\mathbf{h}^{\top}\space\space\in \mathbb{R}^{m\times n},
\end{aligned}
$$

where the standard $k^{th}$ basis vector is $\mathbf{e}_{k}$, the partial derivative w.r.t layer inputs

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial h_{j}} &=\text{w}_{kj},\\
\Rightarrow \frac{\partial \mathbf{g}}{\partial \mathbf{h}} &=\mathbf{W}\space\space\in \mathbb{R}^{m\times n},
\end{aligned}
$$

partial derivative w.r.t intercept

$$
\begin{aligned}
\frac{\partial \text{g}_{k}}{\partial b_{i}} &=\delta_{ki},\\
\Rightarrow \frac{\partial \mathbf{g}}{\partial \mathbf{b}} &=\mathbf{I}_{m}\space\space\in \mathbb{R}^{m\times m},
\end{aligned}
$$

and partial derivative of the piecewise linear function is,

$$
\begin{aligned}
\frac{\partial \text{h}_{j}}{\partial g_{k}} &=\delta_{jk}1_{\text{g}_{k}>0},\\
\Rightarrow \frac{\partial \mathbf{h}}{\partial \mathbf{g}} &=\mathrm{diag}(\mathbf{1}_{\text{g}>0})\space\space\in \mathbb{R}^{m\times m}.
\end{aligned}
$$

If the loss function is a squared loss and the output of the neural network after a forward pass is $f_{\Theta}(\mathbf{x}_d)=\mathbf{h}^{(L)}=\mathbf{f}_d$, then the empirical risk minimisation objective for a sample $(\mathbf{x}_d,\mathbf{y}_d)$ is,

$$
\begin{aligned}
\mathcal{L}_{d}&=\frac{1}{2}{||f_{\Theta}(\mathbf{x}_{d})-\mathbf{y}_{d}||}^{2}_{2},\\
&=
\frac{1}{2}{(\mathbf{f}_{d}-\mathbf{y}_{d})}^{\top}(\mathbf{f}_{d}-\mathbf{y}_{d}),\\
&=\frac{1}{2}\mathbf{f}_{d}^{\top}\mathbf{f}_{d}-2\mathbf{y}_{d}^{\top}\mathbf{f}_{d}+\mathbf{y}_{d}^{\top}\mathbf{y}_{d},\\
\Rightarrow \frac{\partial \mathcal{L}_{d}}{\partial \mathbf{f}_{d}}&=\mathbf{f}_{d}^{\top}-\mathbf{y}_{d}^{\top}\space\space\in \mathbb{R}^{1\times\mathrm{dim}({\mathcal{Y}})}.
\end{aligned}
$$

### Putting it all Together

For each sample $(\mathbf{x}_d,\mathbf{y}_d)$, the network and loss function are evaluated in a forward pass. All of the partial derivative tensors are then calculated and multiplied together to get weight/bias updates. Each weight update is evaluated for a mini-batch using stochastic gradient descent. All of the partial derivative tensors are two-dimensional, apart from the three-dimensional weight partial derivatives, $\frac{\partial \mathbf{g}}{\partial \mathbf{W}}$. In this case, the matrix-tensor multiplication operates along the tensor's third dimension ($k$-index),

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{g}^{(l)}}\frac{\partial \mathbf{g}^{(l)}}{\partial \mathbf{W}^{(l)}},\\
&=\sum_{k}\frac{\partial \mathcal{L}}{\partial \text{g}_{k}^{(l)}}\frac{\partial \text{g}_{k}^{(l)}}{\partial \mathbf{W}^{(l)}}\space\space\in \mathbb{R}^{m{\times} n}.
\end{aligned}
$$
