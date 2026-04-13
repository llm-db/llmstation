---
name: pip-req
description: "Add Python packages to requirements.txt instead of running pip install directly. Use whenever the user wants to install Python packages."
argument-hint: "[package1] [package2] ..."
allowed-tools: Bash, Read, Edit, Write, WebSearch
---

When the user wants to install Python packages, do NOT run `pip install` directly. Instead, add them to `requirements.txt` in the project root.

## Steps

1. **Resolve latest stable versions**: For each package in `$ARGUMENTS`, search PyPI or use `pip index versions <package>` to find the latest stable (non-nightly, non-rc) version.

2. **Read existing requirements.txt**: Read `requirements.txt` if it exists. If it doesn't, create one.

3. **Update requirements.txt**:
   - If the package already exists, update its version.
   - If the package is new, append it.
   - Always pin exact versions with `==`.

4. **Report**: Show the user what was added/updated, and remind them to run:
   ```
   pip install -r requirements.txt
   ```

## Important

- NEVER run `pip install` directly.
- Always pin exact versions (`==`), never use `>=` or unpinned.
- Skip pre-release, nightly, rc, alpha, beta versions.
- Keep existing packages in requirements.txt unchanged unless explicitly asked to update them.
