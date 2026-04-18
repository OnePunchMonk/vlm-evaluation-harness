# VLM-Harness

A unified evaluation framework for Vision Language Models.

See `idea.md` for the full design document.

## Quick Start

```bash
pip install -e ".[anthropic]"
vlm-harness eval --model anthropic:claude-opus-4-6 --bench mmmu --split validation
```
