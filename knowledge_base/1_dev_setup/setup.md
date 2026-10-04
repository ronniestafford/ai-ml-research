# Dev Environment Setup

## Project Structure
[Devcontainers](/.devcontainer) are available for both [python](/.devcontainer/ai-ml-research-python/) and [LaTeX](/.devcontainer/ai-ml-research-python/). The LaTeX container ships with the full LaTeX-Workshop (large container) to compile and render [theory_snippets](/knowledge_base/theory_snippets.tex). Experiments and algorithms are contained in the [src](/src/) folder.

## Native (non DevContainer) Development
If you want to exploit Apple Metal GPU acceleration, you have to develop natively and not in a container. Therefore install [uv](https://docs.astral.sh/uv/getting-started/installation/) directly on your dev machine to manage python and project dependencies,
```bash
$ brew install uv 
```

## Agents
[Opencode](https://opencode.ai/) is provided in each devcontainer for isolation and deep customisation. Or instead, for lightweight isolation, run it in [Docker sandbox](https://docs.docker.com/ai/sandboxes/get-started/) - follow for native installation instructions. This will allow you to achieve isolation even if you are not using devcontainers.

#### Example: OpenCode + llama-server
```bash
$ cd ${PROJECT_ROOT}
$ sbx policy allow network localhost:8080 # e.g. for local llama-server 
$ sbx run opencode
```

Installation instructions of [llama.cpp](https://llama.app/) - see also [repo](https://github.com/ggml-org/llama.cpp). Run llama.cpp with a model, ensuring it binds to the 
```bash
$ llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -c 131072 --jinja --host ::
```

Opencode configuration is provided [here](.opencode/opencode.json) which routes agent requests to `http://host.docker.internal:8080/v1`, as per docker containerised networking. On MacOS the docker sbx proxy forwards requests to IPv6 localhost `[::1]:8080`. Therefore make sure llama-server is listening on the IPv6 equivalent `::1`.
```bash
$llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -c 131072 --jinja --host ::1 
```
