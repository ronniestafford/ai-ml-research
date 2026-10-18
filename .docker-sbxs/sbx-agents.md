# SBX-Agents

Docker sandbox is used as a lightweight isolation solution for coding agents. 

## llama.cpp
The coding agents described below use [llama.cpp](https://llama.app/) to serve LLM model request on the host machine - see also Github [repo](https://github.com/ggml-org/llama.cpp). Once installed, run llama.cpp with a model, ensuring it binds to localhost `::1` (IPv6),
```bash
$ llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -c 131072 --jinja --host ::1 
```

## Docker Sandbox Agents (OpenCode)
[Opencode](https://opencode.ai/) agents are run in [Docker sandbox](https://docs.docker.com/ai/sandboxes/get-started/) to achieve isolation from the host - see host [https://docs.docker.com/ai/sandboxes/install/](installation instructions). You can use other agents, but the following describes running an OpenCode sandbox with a local llama.cpp server running on the host machine.

### Example 1: Out of the Box SBX (OpenCode + llama-server)

Note that the following example doesn't isolate the agent's python virtual environment from the host. See example 2 for recommended full isolation.

```bash
$ cd ${PROJECT_ROOT}
$ sbx policy allow network localhost:8080 # e.g. for local llama-server listening on port 8080.
$ sbx run opencode
```

Opencode configuration is provided [here](.opencode/opencode.json) which routes agent requests to `http://host.docker.internal:8080/v1`, as per docker containerised networking. On MacOS the docker sbx proxy forwards requests to IPv6 localhost `[::1]:8080`.

### Example 2: Using Custom Docker Sandboxes for better Agent Isolation (Recommended)
   - From the host in the repository root, run 
      ```bash
      $ sbx env run ./.docker-sbxs --env-arg workspace_dir="$PWD"
      ``` 
to run customised opencode agent that has its own python environment.
   - The agent is blocked from changing the host's .venv.
