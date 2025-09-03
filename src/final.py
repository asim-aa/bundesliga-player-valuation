#!/usr/bin/env python3
# Thin wrapper that delegates to src/cli.py for predicting
# Bundesliga player market-value progression. Prefer running cli.py directly.
# Usage example:
#   python src/final.py "Jamal Musiala" --years 5 --freq Y

from cli import main

if __name__ == '__main__':
    main()
