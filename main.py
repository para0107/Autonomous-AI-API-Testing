#!/usr/bin/env python3
import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import settings
from core.engine import CoreEngine, APITestRequest
from core.pipeline import TestGenerationPipeline
from input_processing import InputProcessor
from llm import LlamaOrchestrator
from rag.rag_system import RAGSystem
from reinforcement_learning import RLOptimizer
from output.report_generator import ReportGenerator
from feedback.feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


class InputLoader:
    SPEC_EXTENSIONS = {'.json', '.yaml', '.yml'}

    @staticmethod
    def load_spec_file(spec_path: str) -> Dict[str, Any]:
        path = Path(spec_path)
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_path}")
        if path.suffix not in InputLoader.SPEC_EXTENSIONS:
            raise ValueError(f"Unsupported spec format: {path.suffix}")
        with open(path) as f:
            if path.suffix == '.json':
                return json.load(f)
            import yaml
            return yaml.safe_load(f)

    @staticmethod
    def load_source_files(source_dir: str) -> List[str]:
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        extensions = {'.py', '.cs', '.java', '.ts', '.js', '.go', '.rb'}
        files = []
        for ext in extensions:
            files.extend(str(p) for p in source_path.rglob(f'*{ext}'))
        if not files:
            raise ValueError(f"No source files found in: {source_dir}")
        return files

    @staticmethod
    def detect_language(files: List[str]) -> str:
        ext_map = {
            '.py': 'python', '.cs': 'csharp', '.java': 'java',
            '.ts': 'typescript', '.js': 'javascript',
            '.go': 'go', '.rb': 'ruby',
        }
        counts: Dict[str, int] = {}
        for f in files:
            ext = Path(f).suffix
            lang = ext_map.get(ext)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        return max(counts, key=counts.get) if counts else 'python'


def _detect_language(files: List[str]) -> str:
    return InputLoader.detect_language(files)


class Session:
    def __init__(self):
        self.api_spec: Dict[str, Any] = {}
        self.test_cases: List[Dict[str, Any]] = []
        self.execution_results: List[Dict[str, Any]] = []
        self.parsed_spec: Dict[str, Any] = {}
        self.code_files: List[str] = []
        self.language: str = ''
        self.report: Dict[str, Any] = {}
        self.start_time: datetime = datetime.now()


class APITestingAgent:
    def __init__(self):
        self.session = Session()
        self.report_generator = ReportGenerator()

    async def run(self, args: argparse.Namespace) -> int:
        self._configure_logging(args.verbose)
        logger.info("API Testing Agent starting")

        try:
            await self._step1_load_input(args)
            await self._step2_process_input(args)

            if not args.generate_only:
                await self._step3_generate_tests(args)
                await self._step4_execute_tests(args)
                await self._step5_process_feedback(args)
            else:
                await self._step3_generate_tests(args)

            await self._step6_generate_report(args)
            self._print_summary()
            return 0

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=args.verbose)
            return 1

    @staticmethod
    def _configure_logging(verbose: bool):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        )

    async def _step1_load_input(self, args: argparse.Namespace):
        logger.info("STEP 1: Loading input")
        if args.spec:
            self.session.api_spec = InputLoader.load_spec_file(args.spec)
            logger.info(f"Loaded spec: {args.spec}")
        elif args.source:
            files = InputLoader.load_source_files(args.source)
            language = InputLoader.detect_language(files)
            self.session.api_spec = {
                'type': 'source_code',
                'files': {f: open(f).read() for f in files},
                'language': language,
                'root_path': str(Path(args.source).resolve()),
            }
            logger.info(f"Loaded {len(files)} source files ({language})")
        else:
            raise ValueError("Either --spec or --source must be provided")

    async def _step2_process_input(self, args: argparse.Namespace):
        logger.info("STEP 2: Processing input")
        processor = InputProcessor()
        api_type = self.session.api_spec.get('type', 'openapi')

        if api_type == 'source_code':
            files_dict = self.session.api_spec.get('files', {})
            abs_files = list(files_dict.keys())
            language = self.session.api_spec.get('language', 'python')

            self.session.code_files = abs_files
            self.session.language = language

            parsed = processor.parse_code(abs_files, language)
        else:
            spec_file = args.spec or ''
            self.session.code_files = [spec_file]
            self.session.language = 'openapi'

            parsed = processor.parse_code([spec_file], 'openapi')

        if 'endpoints' in parsed and 'results' not in parsed:
            wrapped = {'results': [parsed]}
        else:
            wrapped = parsed

        self.session.parsed_spec = processor.build_specification(wrapped)
        logger.info(
            f"Parsed spec: {len(self.session.parsed_spec.get('endpoints', []))} endpoints"
        )

    async def _step3_generate_tests(self, args: argparse.Namespace):
        logger.info("STEP 3: Generating test cases")
        pipeline = TestGenerationPipeline()
        request_dict = {
            'code_files': self.session.code_files,
            'language': self.session.language,
            'endpoint_url': args.base_url,
            'max_tests': args.max_tests,
            'include_edge_cases': True,
            'auth_token': getattr(args, 'auth_token', None),
            'use_ssl': getattr(args, 'use_ssl', False),
        }
        result = await pipeline.run(request_dict)
        if result.get('status') == 'error':
            raise RuntimeError(f"Pipeline failed: {result.get('error')}")
        self.session.test_cases = (
            result.get('results', {}).get('test_cases') or []
        )
        logger.info(f"Generated {len(self.session.test_cases)} test cases")

    async def _step4_execute_tests(self, args: argparse.Namespace):
        logger.info("STEP 4: Executing test cases")
        engine = CoreEngine()
        request = APITestRequest(
            code_files=self.session.code_files,
            language=self.session.language,
            endpoint_url=args.base_url,
            auth_token=getattr(args, 'auth_token', None),
            use_ssl=getattr(args, 'use_ssl', False),
        )
        self.session.execution_results = await engine._execute_tests(
            self.session.test_cases,
            request,
        )
        passed = sum(1 for r in self.session.execution_results if r.get('passed'))
        logger.info(
            f"Execution complete: {passed}/{len(self.session.execution_results)} passed"
        )

    async def _step5_process_feedback(self, args: argparse.Namespace):
        logger.info("STEP 5: Processing feedback")
        rag = RAGSystem()
        rl = RLOptimizer()
        feedback_loop = FeedbackLoop()
        feedback_loop.set_rag_system(rag)
        feedback_loop.set_rl_optimizer(rl)
        feedback_loop.set_knowledge_base(rag.knowledge_base)
        summary = await feedback_loop.process_feedback(self.session.execution_results)
        logger.info(f"Feedback summary: {summary}")

    async def _step6_generate_report(self, args: argparse.Namespace):
        logger.info("STEP 6: Generating report")
        report = await self.report_generator.generate(
            self.session.execution_results,
            self.session.parsed_spec,
        )
        self.session.report = report

        output_dir = Path(args.output) if args.output else Path('reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f'report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")

    def _print_summary(self):
        results = self.session.execution_results
        total = len(results)
        passed = sum(1 for r in results if r.get('passed'))
        duration = (datetime.now() - self.session.start_time).total_seconds()
        print(f"\n{'=' * 50}")
        print(f"Tests executed : {total}")
        print(f"Tests passed   : {passed}")
        print(f"Tests failed   : {total - passed}")
        print(f"Pass rate      : {passed / total * 100:.1f}%" if total else "Pass rate      : N/A")
        print(f"Duration       : {duration:.1f}s")
        print(f"{'=' * 50}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='API Testing Agent — LLM-powered API test generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--spec', type=str, help='Path to OpenAPI/Swagger spec')
    input_group.add_argument('--source', type=str, help='Path to source code directory')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000')
    parser.add_argument('--output', type=str, default='reports')
    parser.add_argument('--test-types', nargs='+',
                        default=['functional', 'edge_case', 'security'])
    parser.add_argument('--max-tests', type=int, default=50)
    parser.add_argument('--auth-token', type=str, default=None)
    parser.add_argument('--auth-header', type=str, default='Authorization')
    parser.add_argument('--rl-checkpoint', type=str, default=None)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--use-ssl', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser


async def async_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    agent = APITestingAgent()
    return await agent.run(args)


def main():
    sys.exit(asyncio.run(async_main()))


if __name__ == '__main__':
    main()
