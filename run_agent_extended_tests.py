"""
API Testing Agent - Complete System with Full Pipeline
Uses CoreEngine and TestGenerationPipeline for full RAG + RL + 5 Agents + Feedback
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from core.pipeline import TestGenerationPipeline
from core.engine import CoreEngine, APITestRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    print("\n" + "=" * 80)
    print("API Testing Agent - Complete System (RAG + RL + 5 Agents + Feedback)")
    print("=" * 80)
    print("Components:")
    print("  - 5 LLM Agents: Analyzer | TestDesigner | EdgeCase | DataGenerator | ReportWriter")
    print("  - RAG System: Vector Store + Knowledge Base + Retrieval")
    print("  - RL Optimizer: PPO-based test selection and prioritization")
    print("  - Feedback Loop: Continuous learning from execution results")
    print("=" * 80 + "\n")


def get_user_input():
    """Get configuration from user"""
    print("Configuration:")
    print("-" * 80)

    file_path = input("Enter API file path: ").strip()
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return None

    api_url = input("Enter API base URL: ").strip()
    if not api_url:
        print("Error: API URL is required")
        return None

    # Optional: Language detection
    ext_map = {'.cs': 'csharp', '.py': 'python', '.java': 'java', '.cpp': 'cpp'}
    language = ext_map.get(Path(file_path).suffix.lower(), 'csharp')
    print(f"Detected language: {language}")

    # Optional: Authentication
    auth_required = input("Requires authentication? (y/n): ").strip().lower()
    auth_token = None
    if auth_required == 'y':
        auth_token = input("Enter Bearer token: ").strip()

    # Optional: SSL verification
    ssl_verify = input("Use SSL verification? (y/n): ").strip().lower() == 'y'

    # Optional: Max tests
    try:
        max_tests = int(input("Max tests per endpoint (default: 50): ").strip() or "50")
    except ValueError:
        max_tests = 50

    print("-" * 80 + "\n")

    return {
        'file_path': file_path,
        'api_url': api_url,
        'language': language,
        'auth_token': auth_token,
        'ssl_verify': ssl_verify,
        'max_tests': max_tests
    }


def print_results_summary(result: dict):
    """Print detailed results summary"""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)

    if result.get('status') == 'success':
        print("Pipeline completed successfully!\n")

        # Pipeline metrics
        metrics = result.get('metrics', {})
        print("Pipeline Metrics:")
        print(f"  - Total Duration: {metrics.get('total_duration', 0):.2f}s")
        print(f"  - Stages Completed: {metrics.get('stages_completed', 0)}/{metrics.get('total_stages', 0)}")
        print(f"  - Success Rate: {metrics.get('success_rate', 0):.1%}")

        # Test generation metrics
        results = result.get('results', {})
        test_cases = results.get('test_cases', [])
        print(f"\nTest Generation:")
        print(f"  - Tests Generated: {len(test_cases)}")
        print(f"  - Tests Optimized by RL: {'Yes' if 'optimization' in result.get('stages_completed', []) else 'No'}")

        # Execution metrics
        execution_results = results.get('execution_results', [])
        if execution_results:
            passed = sum(1 for r in execution_results if r.get('passed'))
            failed = len(execution_results) - passed
            pass_rate = (passed / len(execution_results) * 100) if execution_results else 0

            print(f"\nTest Execution:")
            print(f"  - Total Tests Executed: {len(execution_results)}")
            print(f"  - Passed: {passed}")
            print(f"  - Failed: {failed}")
            print(f"  - Pass Rate: {pass_rate:.1f}%")

            # Show test type breakdown
            test_types = {}
            for test in test_cases:
                test_type = test.get('test_type', 'unknown')
                test_types[test_type] = test_types.get(test_type, 0) + 1

            if test_types:
                print(f"\nTest Type Distribution:")
                for test_type, count in sorted(test_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {test_type}: {count}")

        # API specification
        api_spec = results.get('api_specification', {})
        if api_spec:
            endpoints = api_spec.get('endpoints', [])
            print(f"\nAPI Analysis:")
            print(f"  - Endpoints Discovered: {len(endpoints)}")
            print(f"  - Validation Rules Found: {len(api_spec.get('validation_rules', []))}")
            print(f"  - Models Extracted: {len(api_spec.get('models', []))}")

        # Report summary
        report = results.get('report', {})
        if report and isinstance(report, dict):
            if report.get('summary'):
                print(f"\nAI Analysis Summary:")
                summary_text = report['summary'][:300]
                print(f"  {summary_text}...")

            if report.get('recommendations'):
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(report['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")

        # Feedback loop status
        if 'feedback' in result.get('stages_completed', []):
            print(f"\nFeedback Loop: Knowledge base and RL model updated")

        print("=" * 80 + "\n")

    else:
        print("Pipeline failed!")
        print("=" * 80)
        error = result.get('error', 'Unknown error')
        print(f"Error: {error}")

        # Show which stages completed before failure
        stages_completed = result.get('stages_completed', [])
        if stages_completed:
            print(f"\nStages completed before failure:")
            for stage in stages_completed:
                print(f"   {stage}")

        print("=" * 80 + "\n")


async def run_with_pipeline(config: dict):
    """Run complete pipeline with all components"""
    logger.info("Initializing Complete Pipeline System...")

    try:
        # Initialize pipeline
        pipeline = TestGenerationPipeline()

        # Prepare request
        request = {
            'code_files': [config['file_path']],
            'language': config['language'],
            'endpoint_url': config['api_url'],
            'max_tests': config.get('max_tests', 50),
            'include_edge_cases': True,
            'auth_token': config.get('auth_token'),
            'use_ssl': config.get('ssl_verify', False)
        }

        logger.info("Starting pipeline execution...")
        logger.info(f"  File: {Path(config['file_path']).name}")
        logger.info(f"  Language: {config['language']}")
        logger.info(f"  API URL: {config['api_url']}")
        logger.info(f"  Max Tests: {config.get('max_tests', 50)}")

        print("\n" + "-" * 80)
        print("Pipeline Stages:")
        print("-" * 80)

        # Run complete pipeline (all 9 stages)
        result = await pipeline.run(request)

        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'stages_completed': []
        }


async def run_with_engine(config: dict):
    """Alternative: Run with CoreEngine (more control)"""
    logger.info("Initializing Core Engine...")

    try:
        # Initialize engine
        engine = CoreEngine()

        # Create API test request
        request = APITestRequest(
            code_files=[config['file_path']],
            language=config['language'],
            endpoint_url=config['api_url'],
            test_types=None,
            max_tests=config.get('max_tests', 50),
            include_edge_cases=True
        )

        logger.info("Starting CoreEngine processing...")

        # Process API (runs full 7-step pipeline in engine.py)
        result = await engine.process_api(request)

        return result

    except Exception as e:
        logger.error(f"CoreEngine execution failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


async def main():
    """Main entry point"""
    print_banner()

    # Get user configuration
    config = get_user_input()
    if not config:
        return

    # Choose execution method
    print("Execution Method:")
    print("  1. TestGenerationPipeline (recommended - 9 stages with full control)")
    print("  2. CoreEngine (alternative - 7-step process)")
    choice = input("Choose method (1/2, default: 1): ").strip() or "1"
    print()

    # Run pipeline
    start_time = datetime.now()

    if choice == "2":
        result = await run_with_engine(config)
    else:
        result = await run_with_pipeline(config)

    # Print results
    print_results_summary(result)

    # Print timing
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal execution time: {duration:.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())