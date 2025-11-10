# `qwen3-4b-instruct-chat-cli`

An example command-line interface (CLI) for interacting with the Qwen 3.4B Instruct Chat model, enabling users to engage in conversational AI tasks directly from the terminal.

<img width="1988" height="1484" alt="image" src="https://github.com/user-attachments/assets/59550fda-0916-45e9-908e-169a1a692050" />


## Installation and execution

1. First, install the required packages and set up a virtual environment using `uv`:

```sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv add transformers
```

2. Run and engage with the AI model! >_<

```sh
python3 main.py
```
