# Contributing

Thanks for contributing to `Ogaal CTC`.

## Development Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `ffmpeg` is available on your system path.

## Local Checks

Run the lightweight repository check before opening a pull request:

```bash
make check
```

## Pull Request Guidelines

- keep changes focused and easy to review
- update docs when user-facing behavior changes
- do not commit model weights, decoder binaries, or local output folders
- keep repository copy aligned with the public Ogaal Labs release wording

## Repository Areas

- `scripts/`: inference CLI, browser demo, and Hugging Face upload helper
- `docs/`: public documentation and figures
- `metadata/`: release metadata for publication tracking
