$$
\providecommand{\mat}[1]{\mathbf{#1}}
\providecommand{\vctr}[1]{\mathbf{#1}}
\providecommand{\spaces}[1]{\mathcal{#1}}
\providecommand{\numberset}[1]{\mathbb{#1}}
\providecommand{\sml}[1]{\,{#1}\,}
\providecommand{\eqs}{\sml{=}}
\providecommand{\defs}{{\,{:}{=}\,}}
\providecommand{\mapdefs}{{\,{:}\,}}
\providecommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\providecommand{\from}{{\leftarrow}}
\providecommand{\tos}{{\to}}
\providecommand{\ins}{\sml{\in}}
\providecommand{\sims}{{\sim}}
\providecommand{\timess}{{\times}}
\providecommand{\imp}{\Rightarrow}
\providecommand{\ele}[1]{\mathrm{#1}}
\providecommand{\twospace}{\space\space}
\providecommand{\argmin}{\mathrm{arg\,min}}
\providecommand{\diag}{\mathrm{diag}}
\providecommand{\dim}{\mathrm{dim}}
\providecommand{\argminu}[2]{{\underset{#1}\argmin{[#2]}}}
\providecommand{\bigargminu}[2]{{\underset{#1}\argmin{\big[#2\big]}}}
\providecommand{\biggargminu}[2]{{\underset{#1}\argmin{\bigg[#2\bigg]}}}
$$

# Backprop

## Risk Minimisation

Assume a joint distribution $P_{X,Y}$ over the input space $\spaces{X}$, output space $\spaces{Y}$, function space $\spaces{F}$ and loss function $\spaces{L}$, population risk $\spaces{R}$, then the general form of the minimisation objective is,

$$
\begin{aligned}
f^{*}\eqs&\argminu{f\in\spaces{F}}{\spaces{R}(f)}\\
\eqs&\biggargminu{f\in\spaces{F}}{\numberset{E}_{P_{X,Y}}\left[ \spaces{L}(f,X,Y) \right]}.
\end{aligned}
$$

The solution $f$ and what it represents depends on i) the form of $P_{X,Y}$ (e.g. its factorisation), ii) function space $\spaces{F}$ and iii) choice of loss function $\spaces{L}$.

## Supervised Learning

If the joint distribution is factorised $P_{X,Y}\eqs P_{X}P_{Y|X}$,
$f\mapdefs\spaces{X}\tos\spaces{Y}$ and loss function is chosen as $\spaces{L}\mapdefs\spaces{Y}\timess\spaces{Y}\tos\numberset{R}$, then the risk minimisation problem becomes

$$
\begin{aligned}
f^{*}\eqs&\biggargminu{f\in\spaces{F}}{\numberset{E}_{P_{X}}\left[ \numberset{E}_{P_{Y|X}}\left[  \spaces{L}(f(X),Y) \right]\right]}.
\end{aligned}
$$

This is the standard supervised learning problem where $P_{X}$ is the data generator distribution and $f$ is modelling some aspect of the conditional distribution. The aspect of $P_{Y|X}$ that is modelled is determined by the choice of loss function e.g. squared loss determines that $f^{*}$ models the conditional mean $\mu_{Y|X\eqs x}\defs\numberset{E}_{P_{Y|X\eqs x}}[Y]$ under the conditional distribution, similarly cross-entropy targets the conditional distribution $P(Y|X\eqs x)$ itself etc.

Empirical risk $\hat{\spaces{R}}$ minimisation requires a finite sample data set $\{(x_d,y_d)\}_{d\eqs1}^D \overset{\mathrm{i.i.d.}}{\sims} P_{X,Y}$, such that

$$
\begin{aligned}
f^{*}\eqs&\biggargminu{f\in\spaces{F}}{\numberset{E}_{P_{Y|X}}\left[  \spaces{L}(f(X),Y) \right]},\\
\eqs&\biggargminu{f\in\spaces{F}}{\frac{1}{D}\sum_{d\eqs 1}^{D}\spaces{L}(f(x_{d}),y_{d})}.
\end{aligned}
$$

## Chain Rule

A deep neural network $f_{\Theta}\mapdefs\spaces{X}\tos\spaces{Y}$ is a composite function of $L$ layers, where each $l^{th}$ layer is itself a composition of a non-linear function $h^{(l)}(\cdot)$ and affine transformation $g_{\theta^{(l)}}^{l}(\cdot)$. The set of parameters for the entire deep network is $\Theta\defs(\theta^{(L)}, \cdots, \theta^{l}, \cdots \theta^{(1)})$, where $\theta^{(l)}$ parameterises $g^{(l)}$,

$$
\begin{aligned}
f_{\Theta}(x)\defs&(h^{(L)} \circ g^{(L)} \circ\cdots\circ h^{(l)} \circ g^{(l)} \circ\cdots\circ h^{(1)} \circ g^{(1)})(x),
\end{aligned}
$$

Then by the chain rule

$$
\begin{aligned}
\pderiv{f_{\Theta}(x)}{\theta^{(l)}}\eqs&\left(\pderiv{h^{(L)}}{g^{(L)}}\pderiv{g^{(L)}}{h^{(L-1)}}  \cdots \pderiv{h^{(l)}}{g^{(l)}}\pderiv{g^{(l)}}{\theta^{(l)}}\right)(x),
\end{aligned}
$$

where each partial derivative is evaluated after $\vctr{x}$ is propagated in a forward pass through the network.

## Gradient Descent

The gradient update rule for layer $l$ at a particular $x_{d}\ins\spaces{X}$ is

$$
\begin{aligned}
\theta^{(l)} &\from \theta^{(l)}-\frac{\eta}{D}\sum_{d\eqs 1}^{D}\left(\pderiv{\spaces{L}_{d}}{f_{\Theta}}\pderiv{f_{\Theta}}{\theta^{(l)}}\right)(x_{d}),
\end{aligned}
$$

where $\eta>0$ is the learning rate. The process of making a forward pass of $x_{d}$ through the entire network, calculating the loss function $\spaces{L}_{d}$ with the target $y_{d}\ins\spaces{Y}$, then calculating the partial derivative tensors in a backwards pass is known as **backpropagation**. Usually each parameter update is done with a gradient calculated for a minibatch $M{\ll} D$.

## Concrete Example

### Matrix \& Vector Notation

Let vectors $\vctr{x}\ins\spaces{X}$, $\vctr{y}\ins\spaces{Y}$. Then the neural network $f_{\Theta}$ consist of the affine transformation

$$
\begin{aligned}
\vctr{g}^{(l)}&\eqs g(\vctr{h}^{(l-1)}),\\
&\eqs\mat{W}^{(l)}\vctr{h}^{(l-1)}+\vctr{b},
\end{aligned}
$$

and non-linearity (piecewise-linear)

$$
\begin{aligned}
\vctr{h}^{(l)}&\eqs h(\vctr{g}^{(l)}),\\
&\eqs\mathrm{ReLU}(\vctr{g}^{(l)}),\\
&\eqs\max(0,\vctr{g}^{(l)}),
\end{aligned}
$$

where the ReLU acts element-wise $\forall \mathrm{g} \in \vctr{g}$,

$$
\begin{aligned}
\mathrm{max}(0,\mathrm{g})&\eqs\begin{cases}
0, & \mathrm{g} \leq 0\\
\mathrm{g}, & \mathrm{g} > 0
\end{cases},
\end{aligned}
$$

with $\vctr{h}^{(0)}\eqs\vctr{x}\ins\spaces{X}$, $\vctr{h}^{(L)}\eqs\vctr{y}\ins\spaces{Y}$, $\mat{W}^{(l)}\ins\numberset{R}^{m\timess n}$, such that $g\mapdefs\numberset{R}^{n}\tos\numberset{R}^{m}$, $h\mapdefs\numberset{R}^{m}\tos\numberset{R}^{m}$. Note that $\vctr{g}^{l}$ and $\vctr{h}^{l}$ are calculated by propagating $\vctr{x}$ forward through the network. Strictly each layer output dimension $m$ is arbitrarily chosen and $n$ equals the previous layer's $m$.

### Partial Derivative Tensors

The chain rule is now

$$
\begin{aligned}
\pderiv{\spaces{L}_{d}}{\mat{W}^{(l)}}\eqs&\left(\pderiv{\spaces{L}_{d}}{h^{(L)}}\pderiv{h^{(L)}}{g^{(L)}}\pderiv{g^{(L)}}{h^{(L-1)}}  \cdots \pderiv{h^{(l)}}{g^{(l)}}\pderiv{g^{(l)}}{\mat{W}^{(l)}}\right)(\vctr{x}_{d}),
\end{aligned}
$$

where from the previous notation,

$$
\Theta\defs(\mat{W}^{(L)}, \cdots, \mat{W}^{(l)}, \cdots \mat{W}^{(1)}).
$$

Temporarily dropping the layer index $l$ and introducing tensor indexes $i$, $j$, $k$, vector elements $\ele{g}_{k}\ins\vctr{g}$, $\ele{h}_{j}\ins\vctr{h}$, $\ele{w}_{ik}\ins\mat{W}$ and $\ele{b}_{k}\ins\vctr{b}$, then the affine transformation is

$$
\begin{aligned}
\ele{g}_{k} &\eqs \sum_{j}\ele{w}_{kj}\ele{h}_{j} + b_{k}.
\end{aligned}
$$

This leads to the partial derivative w.r.t weights

$$
\begin{aligned}
\pderiv{\ele{g}_{k}}{w_{i,j}} &\eqs \ele{h}_{j}\delta_{ik},\\
\imp \pderiv{\ele{g}_{k}}{\mat{W}} &\eqs \vctr{e}_{k}\vctr{h}^{\top}\twospace\ins\numberset{R}^{m\times n},
\end{aligned}
$$

where the standard $k^{th}$ basis vector $\vctr{e}_{k}$, the partial derivative w.r.t layer inputs

$$
\begin{aligned}
\pderiv{\ele{g}_{k}}{h_{j}} &\eqs \ele{w}_{kj},\\
\imp \pderiv{\vctr{g}}{\vctr{h}} &\eqs \mat{W}\twospace\ins\numberset{R}^{m\times n},
\end{aligned}
$$

partial derivative w.r.t intercept

$$
\begin{aligned}
\pderiv{\ele{g}_{k}}{b_{i}} &\eqs \delta_{ki},\\
\imp \pderiv{\vctr{g}}{\vctr{b}} &\eqs \mat{I}_{m}\twospace\ins\numberset{R}^{m\times m},
\end{aligned}
$$

and partial derivative of the piecewise linear function is,

$$
\begin{aligned}
\pderiv{\ele{h}_{j}}{g_{k}} &\eqs 1_{\ele{g}_{k}>0},\\
\imp \pderiv{\vctr{h}}{\vctr{g}} &\eqs \diag(\vctr{1}_{\ele{g}>0})\twospace\ins\numberset{R}^{m\times m}.
\end{aligned}
$$

If the loss function is a squared loss and the output of the neural network after a forward pass is $f_{\Theta}(\vctr{x}_{d})=\vctr{h}^{L}=\vctr{f}_{d}$, then the empirical risk minimisation objective for a sample $(\vctr{x}_{d},\vctr{y}_d)$ is,

$$
\begin{aligned}
\spaces{L}_{d}&\eqs\frac{1}{2}{||f_{\Theta}(\vctr{x}_{d})-\vctr{y}_{d}||}^{2}_{2},\\
&\eqs
\frac{1}{2}{(\vctr{f}_{d}-\vctr{y}_{d})}^{\top}(\vctr{f}_{d}-\vctr{y}_{d}),\\
&\eqs \frac{1}{2}\vctr{f}_{d}^{\top}\vctr{f}_{d}-2\vctr{y}_{d}^{\top}\vctr{f}_{d}+\vctr{y}_{d}^{\top}\vctr{y}_{d},\\
\imp \pderiv{\spaces{L}_{d}}{\vctr{f}_{d}}&\eqs\vctr{f}_{d}^{\top}-\vctr{y}_{d}^{\top}\twospace\ins\numberset{R}^{1\times\dim({\spaces{Y}})}.
\end{aligned}
$$

### Putting it all Together

For each sample $(\vctr{x}_{d},\vctr{y}_d)$, the network and loss function are evaluated in a forward pass. All of the partial derivative tensors are then calculated and multiplied together to get the weight updates. Each weight update is evaluated for a minibatch in stochastic gradient descent. All of the tensors are 2D apart from the 3D weight partial derivative. In this case, the matrix-tensor multiplication operates along the tensor's third dimension (k),

$$
\begin{aligned}
\pderiv{\spaces{L}}{\mat{W}^{(l)}}&\eqs\pderiv{\spaces{L}}{\vctr{g}^{(l)}}\pderiv{\vctr{g}^{(l)}}{\mat{W}^{(l)}},\\
&\eqs\sum_{k}\pderiv{\spaces{L}}{\ele{g}_{k}^{(l)}}\pderiv{\ele{g}_{k}^{(l)}}{\mat{W}^{(l)}}\twospace\ins\numberset{R}^{m\timess n}.
\end{aligned}
$$
