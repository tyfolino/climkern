 # Contributing to ClimKern

 ## 1. Introduction
 The ClimKern Python package was designed to make calculating radiative feedbacks with kernels simple, intuitive, and reproducible. **We welcome virtually any type of contribution within the scope of ClimKern**: feature additions, bug fixes, enhancements, documentation improvements, etc.

 ## 2. How to Contribute:
1. [Open up a GitHub Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) to document your proposed change (if it's new).
2. [Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
3. Create a feature branch *stemming from* `dev`, which is the current development version of ClimKern. `main` is reserved for releases and is behind `dev`.
4. Make your changes.
5. Run tests locally ([`pytest`](https://docs.pytest.org/en/stable/how-to/usage.html)) to check for errors before pushing.
6. [Submit a pull request (PR) to the `dev` branch.](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

## 3. Coding Guidelines
ClimKern uses [Ruff](https://github.com/astral-sh/ruff) for both linting and formatting, configured in `pyproject.toml` and run via [`pre-commit`](https://pre-commit.com). ([MyPy](https://mypy.readthedocs.io/en/stable/) is also wired up for optional local type checking.) Linting and formatting are **enforced in CI**: the `Lint` workflow runs on every pull request and will fail if your code is not formatted, so please install and run the hooks before pushing.

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

Here are some other considerations when contributing:
- Use meaningful names for functions and variables.
- Include formal docstrings for functions in `frontend.py`.
- Explain logic with inline comments.

## 4. Testing

As of version 1.2.0, automated testing via GitHub Actions is not yet implemented. Please test your code using the built-in test suite and `pytest`.

## 5. Review Process

All proposed changes should be submitted as PRs to the dev branch. A maintainer will review (and may edit) the submission. Only maintainers may approve and merge PRs, although reviews from all contributors are welcome.

## 6. Community

GitHub Issues is the ideal place to discuss proposed changes. For additional assistance, please email Ty Janoski or Ivan Mitevski directly.
