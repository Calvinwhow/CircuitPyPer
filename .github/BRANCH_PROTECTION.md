# Required `main` protection

The `Tests` workflow runs for pull requests targeting `main`, merge queues,
and pushes that reach `main`. GitHub Actions runs after Git receives a commit;
therefore a workflow file alone cannot reject a direct local `git push`.

The canonical `Calvinwhow/CircuitPyPer` repository is configured with the
following `main` protection. Apply the same settings to forks or replacement
remotes in **Settings → Rules → Rulesets**:

- Require a pull request before merging.
- Require status checks to pass: `test-suite` and `notebook-output-check`
  (displayed as `Tests / test-suite` and
  `Notebook Hygiene / notebook-output-check` in the Actions UI).
- Require branches to be up to date before merging.
- Block force pushes and deletions.
- Do not allow bypassing the ruleset, except for an intentional emergency role.

For immediate local feedback, install the version-controlled pre-push hook:

```bash
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit install --hook-type pre-push
```

The commit hook strips notebook outputs before they enter history, and the push
hook runs `python -m pytest -q`. They are conveniences rather than security
boundaries; the GitHub ruleset and required status checks are authoritative.
