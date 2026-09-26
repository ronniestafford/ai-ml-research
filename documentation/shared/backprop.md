$$
\providecommand{\mat}[1]{\mathbf{#1}}
\providecommand{\vctr}[1]{\mathbf{#1}}
\providecommand{\spaces}[1]{\mathcal{#1}}
\providecommand{\numberSet}[1]{\mathbb{#1}}
\providecommand{\smallSpacing}[1]{\,{#1}\,}
\providecommand{\eqs}{\smallSpacing{=}}
\providecommand{\defs}{{\,{:}{=}\,}}
\providecommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\providecommand{\from}{\smallSpacing{\leftarrow}}
\providecommand{\ins}{\smallSpacing{\in}}


$$

# Backprop

## Chain Rule

A deep neural network $F_{\Theta}:\spaces{X}\to\spaces{Y}$ is a composite function of $L$ layers, where each $l^{th}$ layer is itself a composition of a non-linear function $h^{(l)}(\cdot)$ and affine transformation $f_{\theta^{(l)}}^{l}(\cdot)$. The set of parameters for the entire deep network is $\Theta\defs(\theta^{(L)}, \cdots, \theta^{l}, \cdots \theta^{(1)})$ where $\theta^{(l)}$ parameterises $f^{(l)}$,

$$
\begin{aligned}
F_{\Theta}(x)\defs&(h^{(L)} \circ f^{(L)} \circ\cdots\circ h^{(l)} \circ f^{(l)} \circ\cdots\circ h^{(1)} \circ f^{(1)})(x),
\end{aligned}
$$

Then by the chain rule

$$
\begin{aligned}
\pderiv{F_{\Theta}(x)}{\theta^{(l)}}\eqs&\left(\pderiv{h^{(L)}}{f^{(L)}}\pderiv{f^{(L)}}{h^{(L-1)}}  \cdots \pderiv{h^{(l)}}{f^{(l)}}\pderiv{f^{(l)}}{\theta^{(l)}}\right)(x),
\end{aligned}
$$

where each partial derivative is evaluated after $\vctr{x}$ is propagated in a forward pass through the network.

## Gradient Descent

With a loss function $\spaces{L}:\spaces{Y}\to\numberSet{R}$, then the gradient update rule for layer $l$ at a particular $x$ is

$$
\begin{aligned}
\theta^{(l)} &\from \theta^{(l)}-\eta\left(\pderiv{\spaces{L}}{F_{\Theta}}\pderiv{F_{\Theta}}{\theta^{(l)}}\right)(x),
\end{aligned}
$$

where $\eta>0$ is the learning rate. The process of making a forward pass of $\vctr{x}$ through the entire network, then calculating the partial derivative tensors in a backwards pass is known as **backpropagation**.

## Concrete Example

Let the affine transformation be

$$
\begin{aligned}
\vctr{z}^{(l)}&=f(h^{(l-1)}),\\
&=\mat{W}^{(l)}\vctr{h}^{(l-1)}+\vctr{b},
\end{aligned}
$$

and non-linearity be piecewise-linear

$$
\begin{aligned}
\vctr{h}^{(l)}&=h(\vctr{z}^{(l)}),\\
&=\operatorname{ReLU}(\vctr{z}^{(l)}),\\
&=\max(0,\vctr{z}^{(l)}),
\end{aligned}
$$

where the ReLU acts element-wise $\forall z \in \vctr{z}$,

$$
\begin{aligned}
\operatorname{max}(0,z)&=\begin{cases}
0, & z \leq 0\\
z, & z > 0
\end{cases},
\end{aligned}
$$

with $\vctr{h}^{(0)}\ins\spaces{X}$, $\vctr{h}^{(L)}\ins\spaces{Y}$, $\mat{W}^{(l)}\ins\numberSet{R}^{m\times n}$, such that $f:\numberSet{R}^{n}\to\numberSet{R}^{m}$ and $h:\numberSet{R}^{m}\to\numberSet{R}^{m}$. Strictly each layer output dimension $m$ is arbitrarily chosen and $n$ equals the previous layer's $m$.

$$
\begin{aligned}
F_{\Theta}(\vctr{x})\defs&(h^{L} \circ f^{L} \circ\cdots\circ h^{l} \circ f^{l} \circ\cdots\circ h^{1} \circ f^{1})(\vctr{x}),
\end{aligned}
$$

$\Theta\defs(\mat{W}^{L}, \cdots, \mat{W}^{l}, \cdots \mat{W}^{1})$

Then by the chain rule

$$
\begin{aligned}
\pderiv{F_{\Theta}(\vctr{x})}{\vctr{\mat{W}}^{l}}\eqs&\left(\pderiv{h^{L}}{f^{L}}\circ\pderiv{f^{L}}{h^{L-1}}  \circ\cdots\circ \pderiv{h^{l}}{f^{l}}\circ\pderiv{f^{l}}{W^{l}}\right)(\vctr{x}).
\end{aligned}
$$

<!-- $$
\begin{aligned}
\mat{A}\eqs&\begin{pmatrix}
a & b \cr
c & d
\end{pmatrix},\\
p(Y=1|\mat{x},\mat{w})\eqs&\text{sigm}(\mat{w}^{\top}\mat{x}),\\
\eqs&\frac{1}{1+e^{-\mat{w}^{\top}\mat{x}}},\\
\eqs&\frac{e^{\mat{w}^{\top}\mat{x}}}{e^{\mat{w}^{\top}\mat{x}}+1},\\
p(Y=0|\mat{x},\mat{w})\eqs&1-\text{sigm}(\mat{w}^{\top}\mat{x}),\\
\eqs&\frac{1}{e^{\mat{w}^{\top}\mat{x}}+1}.
\end{aligned}
$$ -->
