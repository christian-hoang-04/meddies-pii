#!/usr/bin/env python
"""Split a Meddies Labels JSONL bundle into no-leakage train/validation splits."""

from meddies_pii.training.bioes.data.splits import main

if __name__ == "__main__":
    main()
