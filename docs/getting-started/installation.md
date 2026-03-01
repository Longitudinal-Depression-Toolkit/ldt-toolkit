# Installation

`LDT-Toolkit` is built around two required components:

- `ldt-toolkit` (Python toolkit): the execution engine.
- `ldt` (Go CLI): the no-code interface.

Use both for the full experience.

!!! tip "Highly recommended: use UV"
    We strongly recommend [uv](https://docs.astral.sh/uv/) as a state-of-the-art Python package manager for speed, reproducibility, and environment handling.

## Machine Requirements

You need `uv`/`Python`, `Go`, and `R` on your machine.

=== "macOS"

    ```bash
    # Install package managers and runtimes
    brew update
    brew install python@3.12 uv go r

    # Optional: verify
    python3 --version
    uv --version
    go version
    Rscript --version
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip golang r-base curl
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Optional: verify
    python3 --version
    uv --version
    go version
    Rscript --version
    ```

`R` is required when using LCGA/GMM trajectory tooling via `lcmm`.

## Install Python Toolkit (`ldt-toolkit`)

=== "via UV (recommended)"

    ```bash
    uv add ldt-toolkit
    ```

=== "via Pip"

    ```bash
    pip install ldt-toolkit
    ```

Optional R dependencies for LCGA/GMM:

```bash
Rscript --vanilla -e "repos <- 'https://cloud.r-project.org'; required_packages <- c('lcmm'); missing <- setdiff(required_packages, rownames(installed.packages())); if (length(missing)) install.packages(missing, repos = repos) else message('All required R packages are already installed.')"
```

## Install Go CLI Toolkit (`ldt`)

=== "via Homebrew"

    ```bash
    brew tap Longitudinal-Depression-Toolkit/homebrew-tap
    brew install ldt
    ```

=== "via scoop"

    ```powershell
    scoop bucket add longitudinal-depression-toolkit https://github.com/Longitudinal-Depression-Toolkit/scoop-bucket
    scoop install ldt
    ```

## Launch

```bash
ldt
```

Next step: continue with the [Tutorials](../tutorials/index.md).
