"""
Multi-agent coordinator for LLM agents
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from llm.agents.analyzer_agent import AnalyzerAgent
from llm.agents.edge_case_agent import EdgeCaseAgent
from llm.agents.report_writer import ReportWriterAgent
from llm.agents.test_designer import TestDesignerAgent
from llm.agents.data_generator import DataGeneratorAgent




logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system"""
    ANALYZER = "analyzer"
    TEST_DESIGNER = "test_designer"
    EDGE_CASE = "edge_case"
    DATA_GENERATOR = "data_generator"
    REPORT_WRITER = "report_writer"


@dataclass
class AgentTask:
    """Task for an agent to execute"""
    agent_type: AgentType
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1


class AgentManager:
    """Manages and coordinates multiple LLM agents"""

    def __init__(self):
        logger.info("Initializing Agent Manager")

        # Initialize agents
        self.agents = {
            AgentType.ANALYZER: AnalyzerAgent(),
            AgentType.TEST_DESIGNER: TestDesignerAgent(),
            AgentType.EDGE_CASE: EdgeCaseAgent(),
            AgentType.DATA_GENERATOR: DataGeneratorAgent(),
            AgentType.REPORT_WRITER: ReportWriterAgent()
        }

        # Task queue and results
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.running_tasks = {}

    async def orchestrate(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate multiple agents to generate comprehensive test suite

        Args:
            api_spec: API specification
            context: Retrieved context from RAG

        Returns:
            Complete test suite with all components
        """
        logger.info("Starting agent orchestration")

        # Define task workflow
        tasks = self._create_task_workflow(api_spec, context)

        # Execute tasks
        results = await self._execute_workflow(tasks)

        # Combine results
        test_suite = self._combine_results(results)

        return test_suite

    def _create_task_workflow(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> List[AgentTask]:
        """Create workflow of agent tasks"""

        tasks = [
            # First, analyze the API
            AgentTask(
                agent_type=AgentType.ANALYZER,
                input_data={'api_spec': api_spec, 'context': context},
                priority=1
            ),

            # Then design test cases (depends on analysis)
            AgentTask(
                agent_type=AgentType.TEST_DESIGNER,
                input_data={'api_spec': api_spec, 'context': context},
                dependencies=['analyzer'],
                priority=2
            ),

            # Generate edge cases in parallel
            AgentTask(
                agent_type=AgentType.EDGE_CASE,
                input_data={'api_spec': api_spec, 'context': context},
                dependencies=['analyzer'],
                priority=2
            ),

            # Generate test data (depends on test design)
            AgentTask(
                agent_type=AgentType.DATA_GENERATOR,
                input_data={'api_spec': api_spec},
                dependencies=['test_designer', 'edge_case'],
                priority=3
            )
        ]

        return tasks

    async def _execute_workflow(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """Execute workflow of tasks with dependency management"""

        # Sort tasks by priority
        tasks.sort(key=lambda x: x.priority)

        # Execute tasks with dependency resolution
        for task in tasks:
            # Wait for dependencies
            if task.dependencies:
                await self._wait_for_dependencies(task.dependencies)

            # Execute task
            result = await self._execute_task(task)
            self.results[task.agent_type.value] = result

        return self.results

    async def _execute_task(self, task: AgentTask) -> Any:
        """Execute a single agent task"""
        logger.info(f"Executing task: {task.agent_type.value}")

        agent = self.agents[task.agent_type]

        # Add previous results to input if dependencies exist
        if task.dependencies:
            for dep in task.dependencies:
                if dep in self.results:
                    task.input_data[f'{dep}_results'] = self.results[dep]

        # Execute agent
        result = await agent.execute(task.input_data)

        return result

    async def _wait_for_dependencies(self, dependencies: List[str]):
        """Wait for dependent tasks to complete"""
        while not all(dep in self.results for dep in dependencies):
            await asyncio.sleep(0.1)

    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all agents into final test suite"""

        test_suite = {
            'analysis': results.get('analyzer', {}),
            'test_cases': results.get('test_designer', []),
            'edge_cases': results.get('edge_case', []),
            'test_data': results.get('data_generator', {}),
            'metadata': {
                'total_tests': len(results.get('test_designer', [])) + len(results.get('edge_case', [])),
                'agents_used': list(results.keys())
            }
        }

        return test_suite

    async def execute_single_agent(self, agent_type: AgentType, input_data: Dict[str, Any]) -> Any:
        """Execute a single agent independently"""

        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent = self.agents[agent_type]
        result = await agent.execute(input_data)

        return result