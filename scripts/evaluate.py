# file: scripts/evaluate.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from config import rl_config
from reinforcement_learning.policy_network import PolicyNetwork
from reinforcement_learning.rl_optimizer import RLOptimizer
from test_execution.executor import TestExecutor
from output.report_generator import ReportGenerator


class Evaluator:
    """
    Runs test cases through TestExecutor and optionally generates a report.
    Loads a policy network checkpoint if provided.
    """

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
    ) -> None:
        # Use RLOptimizer directly — it owns the networks internally
        self.optimizer = RLOptimizer()

        if checkpoint:
            self.optimizer.load_checkpoint(str(checkpoint))

    async def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        endpoint_url: str,
    ) -> Dict[str, Any]:
        """
        Execute test cases through TestExecutor.

        Args:
            test_cases: List of test case dicts.
            endpoint_url: Base URL to test against.

        Returns:
            Dict with 'results' list and summary stats.
        """
        async with TestExecutor() as executor:
            results = []
            for test in test_cases:
                try:
                    result = await executor.execute_test(test, endpoint_url)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'test': test,
                        'passed': False,
                        'error': str(e),
                    })

        passed = sum(1 for r in results if r.get('passed', False))
        return {
            'results': results,
            'total': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'pass_rate': passed / len(results) if results else 0.0,
        }

    def evaluate_sync(
        self,
        test_cases: List[Dict[str, Any]],
        endpoint_url: str,
    ) -> Dict[str, Any]:
        """Synchronous wrapper around evaluate() for non-async callers."""
        return asyncio.run(self.evaluate(test_cases, endpoint_url))

    def generate_report(
        self,
        results: Any,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate a report using ReportGenerator.
        """
        out_dir = Path(getattr(rl_config, "reports_dir", "data/reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        target = output_path or (out_dir / "report.html")

        try:
            rg = ReportGenerator()
            if hasattr(rg, "generate"):
                rg.generate(results, target)
                return target
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Report generation failed: {e}")

        return None