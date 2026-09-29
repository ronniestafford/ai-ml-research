# Machine Learning Research

Reproducing some old results and experimenting with the new.

## Setup

### DevContainer Development
[Devcontainers](.devcontainer) are available for both [python](.devcontainer/ai-ml-research-python/) and [LaTeX](.devcontainer/ai-ml-research-python/). The LaTeX container ships with LaTeX-Workshop to compile and render PDFs of the [documentation](##Documentation) `.md` files. Experiments and algorithms are contained in the [src](src) folder.

### Native (non DevContainer) Development
If you want to exploit Apple Metal GPU acceleration, you have to develop natively and not in a container. Therefore install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage python and project dependencies,
```bash
$ brew install uv 
```

### Agents
[Opencode](https://opencode.ai/) is provided in each devcontainer for isolation and deep customisation. Or instead, for lightweight isolation, run it in [Docker sandbox](https://docs.docker.com/ai/sandboxes/get-started/) - follow installation instructions. This will allow you to achieve isolation even if you are not using devcontainers.
```bash
$ cd ${PROJECT_ROOT} 
$ sbx run opencode
```

## Documentation

>[!NOTE]
>This repository uses LaTeX formatted text for mathemtatical notation in markdown `.md` files. GitHub converts this into HTML and MathML markup. Safari correctly renders this markup to intended fonts, but MacOS Chrome appears to have limited support for some `<mi mathvariant=... /mi>` MathML attributes, falling back to plain fonts.

### Deep Learning

- [Backprop](documentation/theory_snippets/backprop.md)

## Revisiting my PhD thesis in Model-Based RL

- In progress...
