import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ScratchPad_Agent import ScratchpadAgent


def main() -> None:
    """Run a mock Q1 financial analysis task with the Scratchpad Agent."""
    test_data = {
        "Q1 Revenue": {
            "Product A": 120,
            "Product B": 85,
            "Product C": 200,
            "Product D": 60,
        },
        "Q1 Cost": {
            "Product A": 45,
            "Product B": 80,
            "Product C": 60,
            "Product D": 55,
        },
        "Last Quarter Revenue": {
            "Product A": 100,
            "Product B": 70,
            "Product C": 190,
            "Product D": 50,
        },
    }

    agent = ScratchpadAgent()
    agent.solve(f"""
Please analyze the following Q1 financial data:
{json.dumps(test_data, ensure_ascii=False, indent=2)}

Please complete:
1. Calculate the profit margin for each product
2. Find the fastest-growing product
3. Identify products with abnormally low profit margins (below 20%)
4. Generate an analysis summary
""")


if __name__ == "__main__":
    main()
