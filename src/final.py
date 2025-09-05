#!/usr/bin/env python3
# Thin wrapper that delegates to src/cli.py for predicting
# Bundesliga player market-value progression.
# Behavior: If run with no arguments, defaults to interactive mode.
# Examples:
#   python src/final.py                 # interactive prompts
#   python src/final.py --interactive   # interactive prompts
#   python src/final.py "Jamal Musiala" --years 5 --freq Y

import sys

def main():
    # If no extra args are provided, default to interactive mode
    if len(sys.argv) == 1:
        sys.argv.append("--interactive")
    from cli import main as cli_main
    cli_main()

if __name__ == '__main__':
    main()
