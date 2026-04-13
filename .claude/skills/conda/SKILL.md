---
name: conda
description: "Manage conda environments: activate, create, list, or delete. Use when the user wants to work with conda environments."
argument-hint: "[activate|create|list|delete] [env-name] [python=version]"
allowed-tools: Bash, AskUserQuestion
---

Manage conda environments based on the user's request.

Parse `$ARGUMENTS` to determine the action. The first word is the subcommand:

- **activate <env-name>** — Activate the specified environment.
- **create <env-name> [python=X.Y]** — Create a new environment. If no python version is given, default to `python=3.11`.
- **list** — List all available conda environments.
- **delete <env-name>** — Delete the specified environment (ask for confirmation first).
- If `$ARGUMENTS` is just an env name (no subcommand keyword), treat it as **activate**.
- If `$ARGUMENTS` is empty, treat it as **list**.

All conda commands must be prefixed with:

```
eval "$(conda shell.bash hook)"
```

## Actions

### activate

```
eval "$(conda shell.bash hook)" && conda activate <env-name>
```

After activation, run `python --version` and `which python` to confirm, and report the result.

### create

```
eval "$(conda shell.bash hook)" && conda create -n <env-name> <python=X.Y> -y
```

After creation, activate the new environment and confirm with `python --version` and `which python`.

### list

```
eval "$(conda shell.bash hook)" && conda info --envs
```

Display the result to the user.

### delete

First ask the user to confirm deletion. Then:

```
eval "$(conda shell.bash hook)" && conda remove -n <env-name> --all -y
```

Report the result to the user.
