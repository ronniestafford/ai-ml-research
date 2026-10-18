# Inherit system users, shells, and opencode execution engines securely
FROM docker/sandbox-templates:opencode-docker

# 1. Install packages required to compile Python from source
USER root

# This resolves hash sum mismatch errors from Ubuntu lib repositories 
RUN printf '%s\n' \
    'Acquire::http::Pipeline-Depth "0";' \
    'Acquire::https::Pipeline-Depth "0";' \
    'Acquire::Retries "5";' \
    > /etc/apt/apt.conf.d/99-no-pipeline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libsqlite3-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Configure uv and install it as the agent user
USER agent
WORKDIR /home/agent

# # Install uv securely using the official installer script
# RUN curl -LsSf https://astral.sh | sh
# ENV PATH="/home/agent/.local/bin:${PATH}"

# Forces uv to build the environment inside the container, completely isolated from the host
ENV UV_PROJECT_ENVIRONMENT="/home/agent/.venv_opencode"
# Forces uv cache to stay isolated inside the container's memory/scratch layer
ENV UV_CACHE_DIR="/home/agent/.uv_cache"

# 3. Copy the local entrypoint script
COPY --chmod=755 entrypoint.sh /home/agent/entrypoint.sh
ENTRYPOINT ["/home/agent/entrypoint.sh"]
