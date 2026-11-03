# Machine Learning Research

Reproducing some old results and experimenting with the new.

## Setup 
- Development envrionment [setup](/knowledge_base/1_dev_setup/setup.md).

## Knowledge Base

>[!NOTE]
>This repository uses LaTeX formatted text for mathemtatical notation in markdown `.md` files. GitHub converts this into HTML and MathML markup. Safari correctly renders this markup to intended fonts, but MacOS Chrome appears to have limited support for some `<mi mathvariant=... /mi>` MathML attributes, falling back to plain fonts.

### Theory Snippets

- [Backprop](knowledge_base/2_theory_snippets/backprop.md)


## Algorithms

Each algorithm contains a `code/` folder with the implementation and a `paper/` folder with the associated paper, scaffolded from one of the LaTeX templates in [`template/paper/`](src/ai_ml_research/rl/template/paper/): George Kour's [arxiv-style](https://github.com/kourgeorge/arxiv-style) preprint style [2] (default) or [The Fin AI's arXiv template](https://github.com/The-FinAI/arxiv-template) [3].

- [Template](src/ai_ml_research/rl/template/), scaffolding for creating new algorithms.
- [Vanilla DQN](src/ai_ml_research/rl/dqn), implementation of DeepMind's famous deep Q-learning paper on a small low dimensional state-space MDP.
- [DPCME](src/ai_ml_research/rl/dpcme/), deep model-based approximate policy iteration algorithm with Bellman residual minimisation to fit the value function, proposed in [section 4.5.1](https://discovery.ucl.ac.uk/id/eprint/10089835/3/Stafford_10089835_thesis.pdf#page=124) of my PhD thesis [1].


## Bibliography
[1]: Stafford, R. (2020). *Improving the Practicality of Model-Based Reinforcement Learning: An Investigation into Scaling up Model-Based Methods in Online Settings*. [PhD thesis](https://discovery.ucl.ac.uk/id/eprint/10089835/3/Stafford_10089835_thesis.pdf), University College London.

[2]: Kour, G. (2020). *arxiv-style: A LaTeX style for preprint publications*. [GitHub repository](https://github.com/kourgeorge/arxiv-style), MIT license. Used as the basis of a paper template in each algorithm's `paper/` folder.

[3]: The Fin AI (2026). *The Fin AI arXiv template*. [GitHub repository](https://github.com/The-FinAI/arxiv-template), MIT license (The Fin AI name and logo excluded from the grant and not used here). Used as the basis of a paper template in each algorithm's `paper/` folder.