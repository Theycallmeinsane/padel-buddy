# Contributing

Thanks for helping. Anyone can propose a change; every change is reviewed by
the maintainer before it is merged.

## How a change gets in

1. Fork the repository and make a branch for your change.
2. Open a pull request against `main`. Say what someone would see before and
   after the change, and how it works.
3. The checks (`engine`, `api` and `web`) run on your pull request. On your
   first pull request they wait until the maintainer lets them run.
4. The maintainer reviews it. It can be merged once they approve it and the
   checks are green. New commits after an approval need approving again.

Small, focused pull requests are reviewed fastest. For a large change, open an
issue first to agree on the approach.

## Running the tests

From the repository root:

```bash
(cd engine && pip install -e ".[dev]" && pytest -q)            # engine
pip install -e ./engine -e "./api[dev]" && (cd api && pytest -q)  # api
(cd web && npm ci && npm test && npm run build)                  # web
```

Person detection needs PyTorch for your platform, installed first.

FFmpeg must be on your PATH for the engine and api tests.

## What never goes in the repository

This repository is public.

- Keys and secrets: they go in a local `.env` (git-ignored; copy `.env.example`).
- Match videos, run outputs and model weights: keep them under `data/` and
  `runs/`, which are git-ignored.
- Footage, photos or names of real people. Tests use synthetic scenes and
  made-up names.

## Style

Write code that reads like the code around it: the same naming, comment
density and idioms. Add or update tests for what you change.

## Security issues

Please don't open a public issue for a vulnerability; see
[SECURITY.md](SECURITY.md).

## Licence

By contributing, you agree that your contribution is licensed under the
project's licence ([LICENSE](LICENSE)).
