#!/usr/bin/env python3
"""
API Testing Agent — Main Entry Point
=====================================
Generates comprehensive test cases for APIs using LLM-powered analysis,
RAG-based knowledge retrieval, and RL-optimized test selection.

Usage Examples:
    # From an OpenAPI/Swagger spec file:
    python main.py --spec openapi.yaml --base-url https://api.example.com

    # From source code (auto-detects language):
    python main.py --source ./src/controllers/ --base-url https://localhost:5000

    # Generate only (no execution):
    python main.py --spec openapi.json --generate-only

    # With all options:
    python main.py --spec openapi.yaml \\
        --base-url https://api.staging.com \\
        --output ./reports \\
        --test-types functional edge_case security \\
        --max-tests 50 \\
        --auth-token "Bearer eyJ..." \\
        --auth-header "Authorization" \\
        --rl-checkpoint checkpoints/rl_model.pt \\
        --verbose
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import Settings
from core.engine import CoreEngine
from core.pipeline import TestGenerationPipeline
from input_processing import InputProcessor
from llm import LlamaOrchestrator
from rag.rag_system import RAGSystem
from reinforcement_learning import RLOptimizer
from output.report_generator import ReportGenerator
from feedback.feedback_loop import FeedbackLoop
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Input Loader — reads API specs from files or source directories
# ═══════════════════════════════════════════════════════════════════════════

class InputLoader:
    """Loads and validates API specifications from various sources."""

    SPEC_EXTENSIONS = {'.json', '.yaml', '.yml'}
    SOURCE_EXTENSIONS = {'.py', '.cs', '.java', '.cpp', '.c', '.h', '.hpp'}

    @staticmethod
    def load_spec_file(path: str) -> Dict[str, Any]:
        """Load an OpenAPI/Swagger specification from JSON or YAML."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding='utf-8')

        if suffix == '.json':
            return json.loads(content)
        elif suffix in ('.yaml', '.yml'):
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML spec files. "
                    "Install it: pip install pyyaml"
                )
        else:
            # Try JSON first, then YAML
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    return yaml.safe_load(content)
                except Exception:
                    raise ValueError(
                        f"Cannot parse {path}. "
                        "Supported formats: JSON (.json), YAML (.yaml, .yml)"
                    )

    @staticmethod
    def load_source_directory(path: str) -> Dict[str, Any]:
        """
        Load API source code files from a directory.
        Returns a dict structure the InputProcessor can work with.
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Source directory not found: {path}")

        source_files = {}
        for ext in InputLoader.SOURCE_EXTENSIONS:
            for file in dir_path.rglob(f'*{ext}'):
                rel_path = str(file.relative_to(dir_path))
                source_files[rel_path] = file.read_text(encoding='utf-8', errors='replace')

        if not source_files:
            raise ValueError(
                f"No source files found in {path}. "
                f"Supported extensions: {', '.join(InputLoader.SOURCE_EXTENSIONS)}"
            )

        logger.info(f"Loaded {len(source_files)} source files from {path}")
        return {
            'type': 'source_code',
            'files': source_files,
            'root_path': str(dir_path),
        }

    @staticmethod
    def load(spec_path: Optional[str] = None,
             source_path: Optional[str] = None) -> Dict[str, Any]:
        """Load API definition from either a spec file or source directory."""
        if spec_path:
            logger.info(f"Loading API spec from: {spec_path}")
            return InputLoader.load_spec_file(spec_path)
        elif source_path:
            logger.info(f"Loading source code from: {source_path}")
            return InputLoader.load_source_directory(source_path)
        else:
            raise ValueError("Either --spec or --source must be provided")


# ═══════════════════════════════════════════════════════════════════════════
# Session — holds all runtime context
# ═══════════════════════════════════════════════════════════════════════════

class Session:
    """Runtime session that holds configuration and state for a single run."""

    def __init__(self, args: argparse.Namespace):
        self.id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.args = args
        self.start_time = datetime.now()
        self.api_spec: Dict[str, Any] = {}
        self.parsed_spec: Dict[str, Any] = {}
        self.test_cases: List[Dict[str, Any]] = []
        self.execution_results: List[Dict[str, Any]] = []
        self.report: Optional[Dict[str, Any]] = None

    @property
    def output_dir(self) -> Path:
        base = Path(self.args.output)
        session_dir = base / f"run_{self.id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @property
    def auth_config(self) -> Optional[Dict[str, str]]:
        """Build auth configuration from CLI args."""
        if self.args.auth_token:
            return {
                'header': self.args.auth_header,
                'token': self.args.auth_token,
            }
        if self.args.auth_env:
            token = os.environ.get(self.args.auth_env)
            if token:
                return {
                    'header': self.args.auth_header,
                    'token': token,
                }
            logger.warning(f"Auth env var '{self.args.auth_env}' not set")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat(),
            'base_url': self.args.base_url,
            'test_types': self.args.test_types,
            'max_tests': self.args.max_tests,
            'generate_only': self.args.generate_only,
            'request': {
                'spec_path': self.args.spec,
                'source_path': self.args.source,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Agent Runner — orchestrates the full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class AgentRunner:
    """
    Top-level orchestrator that wires all components together and runs
    the test generation + execution pipeline.

    This replaces the old run_agent_extended_tests.py with a cleaner,
    parameterized entry point.
    """

    def __init__(self, session: Session):
        self.session = session
        self._components_initialized = False

    async def _init_components(self):
        """Lazy-initialize all pipeline components."""
        if self._components_initialized:
            return

        logger.info("Initializing pipeline components...")

        # Core components
        self.input_processor = InputProcessor()
        self.orchestrator = LlamaOrchestrator()
        self.rag_system = RAGSystem()  # Auto-initializes (Bug 3 fix)
        self.rl_optimizer = RLOptimizer()
        self.report_generator = ReportGenerator()
        self.feedback_loop = FeedbackLoop()

        # Load RL checkpoint if provided
        if self.session.args.rl_checkpoint:
            rl_path = Path(self.session.args.rl_checkpoint)
            if rl_path.exists():
                self.rl_optimizer.load(str(rl_path))
                logger.info(f"Loaded RL model from {rl_path}")

        # Check LLM connectivity
        try:
            connected = await self.orchestrator.check_connection()
            if not connected:
                logger.warning(
                    "LLM connection check failed. "
                    "Ensure the LLM server is running and configured in .env"
                )
        except Exception as e:
            logger.warning(f"LLM connection check error: {e}")

        self._components_initialized = True
        logger.info("All components initialized")

    async def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and return results."""
        await self._init_components()
        args = self.session.args

        # ── Step 1: Load API definition ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Loading API definition")
        logger.info("=" * 60)

        self.session.api_spec = InputLoader.load(
            spec_path=args.spec,
            source_path=args.source,
        )

        # ── Step 2: Parse & extract endpoints ───────────────────────────
        logger.info("STEP 2: Parsing API specification")

        self.session.parsed_spec = await self.input_processor.process(
            self.session.api_spec
        )

        endpoint_count = len(self.session.parsed_spec.get('endpoints', []))
        logger.info(f"Extracted {endpoint_count} endpoints")

        if endpoint_count == 0:
            logger.error("No endpoints found. Check your spec/source files.")
            return {'status': 'error', 'message': 'No endpoints extracted'}

        # Inject base_url into the parsed spec
        if args.base_url:
            self.session.parsed_spec['base_url'] = args.base_url

        # Inject auth config
        auth = self.session.auth_config
        if auth:
            self.session.parsed_spec['auth'] = auth
            logger.info(f"Auth configured via header: {auth['header']}")

        # ── Step 3: Generate test cases ─────────────────────────────────
        logger.info("STEP 3: Generating test cases")

        # Build the pipeline with all components
        pipeline = TestGenerationPipeline(
            orchestrator=self.orchestrator,
            rag_system=self.rag_system,
            rl_optimizer=self.rl_optimizer,
        )

        self.session.test_cases = await pipeline.generate(
            api_spec=self.session.parsed_spec,
            test_types=args.test_types,
            max_tests=args.max_tests,
        )

        logger.info(f"Generated {len(self.session.test_cases)} test cases")

        # Save generated test cases
        tests_file = self.session.output_dir / "test_cases.json"
        with open(tests_file, 'w') as f:
            json.dump(self.session.test_cases, f, indent=2, default=str)
        logger.info(f"Test cases saved to {tests_file}")

        # ── Step 4: Execute tests (unless --generate-only) ──────────────
        if args.generate_only:
            logger.info("--generate-only flag set. Skipping execution.")
            return {
                'status': 'success',
                'mode': 'generate_only',
                'test_cases': len(self.session.test_cases),
                'output_dir': str(self.session.output_dir),
            }

        if not args.base_url:
            logger.warning(
                "No --base-url provided. Cannot execute tests. "
                "Use --generate-only or provide --base-url."
            )
            return {
                'status': 'partial',
                'mode': 'generate_only_no_url',
                'test_cases': len(self.session.test_cases),
                'output_dir': str(self.session.output_dir),
            }

        logger.info("STEP 4: Executing test cases")

        engine = CoreEngine(
            orchestrator=self.orchestrator,
            rag_system=self.rag_system,
            rl_optimizer=self.rl_optimizer,
            report_generator=self.report_generator,
        )

        self.session.execution_results = await engine.execute(
            test_cases=self.session.test_cases,
            api_spec=self.session.parsed_spec,
            base_url=args.base_url,
        )

        # ── Step 5: Generate report ─────────────────────────────────────
        logger.info("STEP 5: Generating report")

        self.session.report = await self.report_generator.generate(
            self.session.execution_results,
            self.session.parsed_spec,
        )

        # Save report
        report_file = self.session.output_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(self.session.report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_file}")

        # ── Step 6: Feedback loop (learn from results) ──────────────────
        if not args.no_feedback:
            logger.info("STEP 6: Running feedback loop")
            try:
                await self.feedback_loop.process(
                    test_cases=self.session.test_cases,
                    execution_results=self.session.execution_results,
                    api_spec=self.session.parsed_spec,
                )
                logger.info("Feedback loop completed")
            except Exception as e:
                logger.warning(f"Feedback loop error (non-fatal): {e}")

        # ── Step 7: Save RL checkpoint ──────────────────────────────────
        if args.rl_checkpoint:
            rl_path = Path(args.rl_checkpoint)
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_optimizer.save(str(rl_path))
            logger.info(f"RL model saved to {rl_path}")

        # ── Summary ─────────────────────────────────────────────────────
        elapsed = (datetime.now() - self.session.start_time).total_seconds()
        summary = _build_summary(self.session, elapsed)
        _print_summary(summary)

        # Save session metadata
        meta_file = self.session.output_dir / "session.json"
        with open(meta_file, 'w') as f:
            json.dump({**self.session.to_dict(), **summary}, f, indent=2, default=str)

        return summary


# ═══════════════════════════════════════════════════════════════════════════
# Summary helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_summary(session: Session, elapsed: float) -> Dict[str, Any]:
    """Build a results summary dict."""
    total = len(session.execution_results)
    passed = sum(1 for r in session.execution_results if r.get('status') == 'passed')
    failed = sum(1 for r in session.execution_results if r.get('status') == 'failed')
    errors = sum(1 for r in session.execution_results if r.get('status') == 'error')

    return {
        'status': 'success',
        'session_id': session.id,
        'elapsed_seconds': round(elapsed, 2),
        'tests_generated': len(session.test_cases),
        'tests_executed': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': f"{100 * passed / total:.1f}%" if total > 0 else "N/A",
        'output_dir': str(session.output_dir),
    }


def _print_summary(summary: Dict[str, Any]):
    """Print a human-readable summary to the console."""
    print("\n" + "=" * 60)
    print("  API TESTING AGENT — RUN COMPLETE")
    print("=" * 60)
    print(f"  Session:        {summary.get('session_id', 'N/A')}")
    print(f"  Duration:       {summary.get('elapsed_seconds', 0)}s")
    print(f"  Tests Generated:{summary.get('tests_generated', 0)}")
    print(f"  Tests Executed: {summary.get('tests_executed', 0)}")
    print(f"  Passed:         {summary.get('passed', 0)}")
    print(f"  Failed:         {summary.get('failed', 0)}")
    print(f"  Errors:         {summary.get('errors', 0)}")
    print(f"  Pass Rate:      {summary.get('pass_rate', 'N/A')}")
    print(f"  Output:         {summary.get('output_dir', '')}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Argument Parser
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='api-testing-agent',
        description=(
            'AI-powered API test generation and execution agent. '
            'Analyzes API specs or source code, generates comprehensive '
            'test cases using LLM + RAG + RL, and optionally executes them.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --spec openapi.yaml --base-url https://api.example.com
  %(prog)s --source ./controllers/ --generate-only
  %(prog)s --spec api.json --base-url http://localhost:8080 --test-types functional edge_case security
  %(prog)s --spec api.yaml --base-url https://staging.api.com --auth-token "Bearer eyJ..." --max-tests 100
        """,
    )

    # ── Input (required: one of --spec or --source) ─────────────────────
    input_group = parser.add_argument_group('Input (required: one of --spec or --source)')
    input_exclusive = input_group.add_mutually_exclusive_group(required=True)
    input_exclusive.add_argument(
        '--spec', type=str, metavar='PATH',
        help='Path to OpenAPI/Swagger spec file (.json, .yaml, .yml)',
    )
    input_exclusive.add_argument(
        '--source', type=str, metavar='DIR',
        help='Path to API source code directory (auto-detects language)',
    )

    # ── Target ──────────────────────────────────────────────────────────
    target_group = parser.add_argument_group('Target API')
    target_group.add_argument(
        '--base-url', type=str, metavar='URL',
        help='Base URL of the API to test (e.g., https://api.example.com)',
    )

    # ── Test Generation ─────────────────────────────────────────────────
    gen_group = parser.add_argument_group('Test Generation')
    gen_group.add_argument(
        '--test-types', nargs='+', metavar='TYPE',
        default=['functional', 'edge_case', 'security', 'boundary', 'negative'],
        help=(
            'Types of tests to generate. '
            'Choices: functional, edge_case, security, performance, '
            'boundary, negative, integration. '
            'Default: functional edge_case security boundary negative'
        ),
    )
    gen_group.add_argument(
        '--max-tests', type=int, default=100, metavar='N',
        help='Maximum number of test cases to generate (default: 100)',
    )
    gen_group.add_argument(
        '--generate-only', action='store_true',
        help='Generate test cases without executing them',
    )
    gen_group.add_argument(
        '--endpoints', nargs='+', metavar='PATTERN',
        help=(
            'Only generate tests for endpoints matching these patterns '
            '(e.g., "/users" "/orders/*")'
        ),
    )

    # ── Authentication ──────────────────────────────────────────────────
    auth_group = parser.add_argument_group('Authentication')
    auth_group.add_argument(
        '--auth-token', type=str, metavar='TOKEN',
        help='Auth token to include in requests (e.g., "Bearer eyJ...")',
    )
    auth_group.add_argument(
        '--auth-env', type=str, metavar='VAR_NAME',
        help='Read auth token from this environment variable',
    )
    auth_group.add_argument(
        '--auth-header', type=str, default='Authorization', metavar='HEADER',
        help='Header name for auth token (default: Authorization)',
    )

    # ── Output ──────────────────────────────────────────────────────────
    out_group = parser.add_argument_group('Output')
    out_group.add_argument(
        '--output', type=str, default='./reports', metavar='DIR',
        help='Output directory for reports and test cases (default: ./reports)',
    )
    out_group.add_argument(
        '--format', type=str, choices=['json', 'qase', 'html'], default='json',
        help='Report format (default: json)',
    )

    # ── RL & Learning ───────────────────────────────────────────────────
    rl_group = parser.add_argument_group('RL & Learning')
    rl_group.add_argument(
        '--rl-checkpoint', type=str, metavar='PATH',
        help='Path to RL model checkpoint (load/save)',
    )
    rl_group.add_argument(
        '--no-feedback', action='store_true',
        help='Skip the feedback loop after execution',
    )

    # ── Runtime ─────────────────────────────────────────────────────────
    runtime_group = parser.add_argument_group('Runtime')
    runtime_group.add_argument(
        '--timeout', type=int, default=30, metavar='SEC',
        help='HTTP request timeout in seconds (default: 30)',
    )
    runtime_group.add_argument(
        '--concurrency', type=int, default=5, metavar='N',
        help='Max concurrent test executions (default: 5)',
    )
    runtime_group.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose/debug logging',
    )
    runtime_group.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all output except errors',
    )
    runtime_group.add_argument(
        '--dry-run', action='store_true',
        help='Validate inputs and show what would be done, without running',
    )

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

async def async_main(args: argparse.Namespace) -> int:
    """Async entry point."""

    # Configure logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    # Dry-run mode
    if args.dry_run:
        print("\n[DRY RUN] Configuration:")
        print(f"  Spec:         {args.spec or '(from source)'}")
        print(f"  Source:       {args.source or '(from spec)'}")
        print(f"  Base URL:     {args.base_url or '(not set — generate-only)'}")
        print(f"  Test types:   {', '.join(args.test_types)}")
        print(f"  Max tests:    {args.max_tests}")
        print(f"  Generate only:{args.generate_only}")
        print(f"  Auth:         {'configured' if (args.auth_token or args.auth_env) else 'none'}")
        print(f"  Output:       {args.output}")
        print(f"  RL checkpoint:{args.rl_checkpoint or 'none'}")
        print(f"  Feedback:     {'disabled' if args.no_feedback else 'enabled'}")
        print(f"  Timeout:      {args.timeout}s")
        print(f"  Concurrency:  {args.concurrency}")

        # Validate the spec/source can be loaded
        try:
            InputLoader.load(spec_path=args.spec, source_path=args.source)
            print("\n  ✓ Input file validated successfully")
        except Exception as e:
            print(f"\n  ✗ Input validation failed: {e}")
            return 1

        print("\n[DRY RUN] Would proceed with test generation. Exiting.")
        return 0

    # Run the agent
    session = Session(args)

    try:
        runner = AgentRunner(session)
        result = await runner.run()

        if result.get('status') == 'error':
            logger.error(f"Run failed: {result.get('message')}")
            return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(async_main(args)))


if __name__ == '__main__':
    main()