"""
Multi-agent coordinator for LLM agents
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

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
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1


class AgentManager:
    """Manages and coordinates multiple LLM agents.

    FIX: Agents require a llama_client. AgentManager must receive one
    or be used via LlamaOrchestrator which provides the client.
    """

    def __init__(self, llama_client=None):
        logger.info("Initializing Agent Manager")

        self._client = llama_client
        self.agents = {}
        self.results = {}

        if llama_client is not None:
            self._init_agents(llama_client)

    def _init_agents(self, llama_client):
        """Initialize agents with the given LLM client."""
        from llm.agents.analyzer_agent import AnalyzerAgent
        from llm.agents.edge_case_agent import EdgeCaseAgent
        from llm.agents.report_writer import ReportWriterAgent
        from llm.agents.test_designer import TestDesignerAgent
        from llm.agents.data_generator import DataGeneratorAgent

        self.agents = {
            AgentType.ANALYZER: AnalyzerAgent(llama_client),
            AgentType.TEST_DESIGNER: TestDesignerAgent(llama_client),
            AgentType.EDGE_CASE: EdgeCaseAgent(llama_client),
            AgentType.DATA_GENERATOR: DataGeneratorAgent(llama_client),
            AgentType.REPORT_WRITER: ReportWriterAgent(llama_client),
        }

    async def orchestrate(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate agents to generate a test suite."""
        if not self.agents:
            raise RuntimeError(
                "AgentManager has no agents. "
                "Provide a llama_client when constructing, or use LlamaOrchestrator."
            )

        logger.info("Starting agent orchestration")
        tasks = self._create_task_workflow(api_spec, context)
        results = await self._execute_workflow(tasks)
        return self._combine_results(results)

    def _create_task_workflow(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> List[AgentTask]:
        """Create workflow of agent tasks"""
        return [
            AgentTask(
                agent_type=AgentType.ANALYZER,
                input_data={'api_spec': api_spec, 'context': context},
                priority=1
            ),
            AgentTask(
                agent_type=AgentType.TEST_DESIGNER,
                input_data={'api_spec': api_spec, 'context': context},
                dependencies=['analyzer'],
                priority=2
            ),
            AgentTask(
                agent_type=AgentType.EDGE_CASE,
                input_data={'api_spec': api_spec, 'context': context},
                dependencies=['analyzer'],
                priority=2
            ),
            AgentTask(
                agent_type=AgentType.DATA_GENERATOR,
                input_data={'api_spec': api_spec},
                dependencies=['test_designer', 'edge_case'],
                priority=3
            ),
        ]

    async def _execute_workflow(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """Execute workflow with dependency management"""
        tasks.sort(key=lambda x: x.priority)

        for task in tasks:
            if task.dependencies:
                await self._wait_for_dependencies(task.dependencies)
            result = await self._execute_task(task)
            self.results[task.agent_type.value] = result

        return self.results

    async def _execute_task(self, task: AgentTask) -> Any:
        """Execute a single agent task"""
        logger.info(f"Executing task: {task.agent_type.value}")
        agent = self.agents[task.agent_type]

        if task.dependencies:
            for dep in task.dependencies:
                if dep in self.results:
                    task.input_data[f'{dep}_results'] = self.results[dep]

        return await agent.execute(task.input_data)

    async def _wait_for_dependencies(self, dependencies: List[str]):
        """Wait for dependent tasks to complete"""
        while not all(dep in self.results for dep in dependencies):
            await asyncio.sleep(0.1)

    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results into final test suite"""
        test_cases = results.get('test_designer', {})
        if isinstance(test_cases, dict):
            test_cases = test_cases.get('happy_path_tests', [])

        edge_cases = results.get('edge_case', [])

        return {
            'analysis': results.get('analyzer', {}),
            'test_cases': test_cases,
            'edge_cases': edge_cases,
            'test_data': results.get('data_generator', {}),
            'metadata': {
                'total_tests': len(test_cases) + len(edge_cases),
                'agents_used': list(results.keys())
            }
        }

    async def execute_single_agent(self, agent_type: AgentType, input_data: Dict[str, Any]) -> Any:
        """Execute a single agent independently"""
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return await self.agents[agent_type].execute(input_data)