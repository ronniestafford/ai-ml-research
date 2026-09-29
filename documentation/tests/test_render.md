# Test Rendering

File to test rendering: GitHub (MathJax), VSCode .md preview and Latex PDF.

## Characters

### Results

TLDR: Latex has trouble rendering unicode, Chrome MacOS has trouble rendering Github .md files. Chrome is being presented with the correct html  

mathcal (Cursive, Spaces)

- GitHub: no (Chrome MacOS) yes (safari).
- Preview: yes.
- Latex: yes.

mathscr (Cursive, Spaces)

- GitHub: no (Chrome MacOS) yes (safari).
- Preview: yes.
- Latex: yes.

mathbf (Bold Vector and Matrices)

- GitHub: no (Chrome MacOS) yes (safari).
- Preview: yes.
- Latex: yes.

mathbb (Number Sets)

- GitHub: no (Chrome MacOS) yes (safari).
- Preview: yes.
- Latex: yes.

text (Elements and Operators)

- GitHub: yes.
- Preview: yes.
- Latex: yes.

mathrm (Elements and Operators)

- GitHub: yes.
- Preview: yes.
- Latex: yes.

### mathcal (Cursive, Spaces)

#### Inline

$\mathcal{L}$,

$\mathcal{X}$, 

$\mathcal{Y}$, 

$\mathcal{F}$, 

$\mathcal{R}$, 

$\mathcal{L}_i$,

$\mathcal{X}_i$,

$\mathcal{Y}_i$,

$\mathcal{F}_i$

$\mathcal{R}_i$

#### Block

$$ \mathcal{L}$$
$$\mathcal{X}$$
$$\mathcal{Y}$$
$$\mathcal{F}$$
$$\mathcal{R}$$

$$\mathcal{L}_i$$
$$\mathcal{X}_i$$
$$\mathcal{Y}_i$$
$$\mathcal{F}_i$$
$$\mathcal{R}_i$$

### mathscr (Cursive, Spaces)

#### Inline

$\mathscr{L}$

$\mathscr{X}$

$\mathscr{Y}$

$\mathscr{F}$

$\mathscr{R}$

$\mathscr{L}_i$

$\mathscr{X}_i$

$\mathscr{Y}_i$

$\mathscr{F}_i$

$\mathscr{R}_i$

#### Block

$$\mathscr{L}$$
$$\mathscr{X}$$
$$\mathscr{Y}$$
$$\mathscr{F}$$
$$\mathscr{R}$$

$$\mathscr{L}_i$$
$$\mathscr{X}_i$$
$$\mathscr{Y}_i$$
$$\mathscr{F}_i$$
$$\mathscr{R}_i$$

### mathbf (Bold Vector and Matrices)
#### Inline

$\mathbf{h}$, 

$\mathbf{W}$, 

$\mathbf{f}$,

$\mathbf{h}_i$

$\mathbf{W}_i$

$\mathbf{f}_i$

#### Block

$$\mathbf{h}$$
$$\mathbf{W}$$
$$\mathbf{f}$$

$$\mathbf{h}_i$$
$$\mathbf{W}_i$$
$$\mathbf{f}_i$$

### text (Elements and Operators)

#### Inline

$\text{h}$,

$\text{W}$,

$\text{f}$,

$\text{h}_i$,

$\text{W}_i$,

$\text{f}_i$,

#### Block

$$\text{h}$$
$$\text{W}$$
$$\text{f}$$

$$\text{h}_i$$
$$\text{W}_i$$
$$\text{f}_i$$

### mathbb (Number Sets)

#### Inline

$\mathbb{E}$, 

$\mathbb{R}$, 

$\mathbb{E}_i$

$\mathbb{R}_i$

#### Block

$$\mathbb{E}$$
$$\mathbb{R}$$

$$\mathbb{E}_i$$
$$\mathbb{R}_i$$

### mathrm (Elements and Operators)

#### Inline

$\mathrm{h}$,

$\mathrm{W}$,

$\mathrm{f}$,

$\mathrm{h}_i$,

$\mathrm{W}_i$,

$\mathrm{f}_i$,

#### Block

$$\mathrm{h}$$
$$\mathrm{W}$$
$$\mathrm{f}$$

$$\mathrm{h}_i$$
$$\mathrm{W}_i$$
$$\mathrm{f}_i$$

### Unicode
#### Inline
 $𝓛$, $𝓧$, $𝒴$, $ℱ$, $R$
 
 $𝐡$, $𝐖$, $𝐟$
 
 $𝔼$, $ℝ$

 #### Block
 $$𝓛, 𝓧, 𝒴, ℱ, R, 𝐡, 𝐖, 𝐟, 𝔼, ℝ$$

### Math blocks test

Inline:

$\mathbb{R}$

Block:

$$
\mathbb{R}
$$

Math block:

```math
\mathbb{R}
```

Other fonts:

$$
\mathbf{R}\quad
\mathcal{R}\quad
\mathscr{R}\quad
\mathfrak{R}
$$