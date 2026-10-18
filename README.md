# Machine Learning Research

Reproducing some old results and experimenting with the new.

## Setup 
- Development envrionment [setup](/knowledge_base/1_dev_setup/setup.md).

## Knowledge Base

>[!NOTE]
>This repository uses LaTeX formatted text for mathemtatical notation in markdown `.md` files. GitHub converts this into HTML and MathML markup. Safari correctly renders this markup to intended fonts, but MacOS Chrome appears to have limited support for some `<mi mathvariant=... /mi>` MathML attributes, falling back to plain fonts.

### Theory Snippets

- [Backprop](knowledge_base/2_theory_snippets/backprop.md)


## Experiments

- [Template](src/ai_ml_research/rl/template/), scaffolding for creating new agents and experiments.
- [Vanilla DQN](src/ai_ml_research/rl/dqn), implementation of DeepMind's famous deep Q-learning paper on a small low dimensional state-space MDP.
- [DPCME](src/ai_ml_research/rl/dpcme/), deep model-based approximate policy iteration algorithm with Bellman residual minimisation to fit the value function proposed in my [1].


## Bibliography
[1]: Stafford, R. (2020). *Improving the Practicality of Model-Based Reinforcement Learning: An Investigation into Scaling up Model-Based Methods in Online Settings*, [Section 4.5.1](https://discovery.ucl.ac.uk/id/eprint/10089835/3/Stafford_10089835_thesis.pdf#page=124) PhD thesis, University College London.