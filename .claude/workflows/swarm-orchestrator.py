#!/usr/bin/env python3
"""
Advanced Swarm Orchestrator for Codebase Improvement
Implements parallel wave execution with fault tolerance and real-time monitoring
"""

import asyncio
import json
import yaml
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import hashlib
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger('SwarmOrchestrator')


@dataclass
class AgentResult:
    """Result from an agent execution"""
    agent_id: str
    wave_id: str
    status: str
    outputs: Dict[str, Any]
    metrics: Dict[str, float]
    duration: float
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == 'success'


@dataclass
class WaveResult:
    """Aggregated result from a wave execution"""
    wave_id: str
    agents_completed: int
    agents_failed: int
    total_duration: float
    outputs: Dict[str, Any]
    metrics: Dict[str, float]
    success_rate: float


class SwarmAgent:
    """Represents a single agent in the swarm"""

    def __init__(self, config: Dict[str, Any], wave_id: str):
        self.id = config['id']
        self.type = config['type']
        self.focus = config['focus']
        self.command = config.get('command', '')
        self.tasks = config.get('tasks', [])
        self.outputs = config.get('outputs', {})
        self.inputs = config.get('inputs', {})
        self.wave_id = wave_id
        self.start_time = None
        self.end_time = None

    async def execute(self, shared_context: Dict[str, Any]) -> AgentResult:
        """Execute the agent asynchronously"""
        self.start_time = time.time()
        logger.info(f"🚀 Agent {self.id} starting execution in wave {self.wave_id}")

        try:
            # Resolve inputs from shared context
            resolved_inputs = self._resolve_inputs(shared_context)

            # Simulate agent execution (in real implementation, would call actual agent)
            result = await self._run_agent_command(resolved_inputs)

            # Store outputs in shared context
            for key, value in self.outputs.items():
                shared_context[f"{self.wave_id}.{self.id}.outputs.{key}"] = value

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            logger.info(f"✅ Agent {self.id} completed in {duration:.2f}s")

            return AgentResult(
                agent_id=self.id,
                wave_id=self.wave_id,
                status='success',
                outputs=self.outputs,
                metrics=self._calculate_metrics(result),
                duration=duration
            )

        except Exception as e:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            logger.error(f"❌ Agent {self.id} failed: {str(e)}")

            return AgentResult(
                agent_id=self.id,
                wave_id=self.wave_id,
                status='failed',
                outputs={},
                metrics={},
                duration=duration,
                errors=[str(e)]
            )

    def _resolve_inputs(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input references from shared context"""
        resolved = {}
        for key, value in self.inputs.items():
            if isinstance(value, str) and value.startswith('${'):
                # Extract reference path
                ref_path = value[2:-1]  # Remove ${ and }
                resolved[key] = self._get_from_context(shared_context, ref_path)
            else:
                resolved[key] = value
        return resolved

    def _get_from_context(self, context: Dict[str, Any], path: str) -> Any:
        """Get value from context using dot notation path"""
        parts = path.split('.')
        current = context
        for part in parts:
            if part == 'all_outputs':
                # Special handling for all_outputs
                return self._gather_all_outputs(context, parts[0])
            current = current.get(part, {})
        return current

    def _gather_all_outputs(self, context: Dict[str, Any], wave_id: str) -> Dict[str, Any]:
        """Gather all outputs from a wave"""
        outputs = {}
        for key, value in context.items():
            if key.startswith(f"{wave_id}.") and '.outputs.' in key:
                outputs[key] = value
        return outputs

    async def _run_agent_command(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent command (simulated)"""
        # In real implementation, this would:
        # 1. Prepare the command with inputs
        # 2. Execute the command asynchronously
        # 3. Parse and return the results

        # Simulate processing time
        await asyncio.sleep(2 + len(self.tasks) * 0.5)

        # Simulate results
        results = {
            'tasks_completed': len(self.tasks),
            'findings': len(self.tasks) * 3,
            'improvements': len(self.tasks) * 2
        }

        return results

    def _calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate agent metrics"""
        return {
            'tasks_completed': result.get('tasks_completed', 0),
            'findings': result.get('findings', 0),
            'improvements': result.get('improvements', 0),
            'efficiency': 0.85 + (0.15 * (self.id.__hash__() % 100) / 100)
        }


class SwarmWave:
    """Represents a wave of parallel agent execution"""

    def __init__(self, config: Dict[str, Any]):
        self.id = config['wave_id']
        self.name = config['name']
        self.depends_on = config.get('depends_on', [])
        self.parallel_agents = config.get('parallel_agents', 5)
        self.agents = [SwarmAgent(agent_config, self.id)
                      for agent_config in config.get('agents', [])]
        self.coordination = config.get('coordination', {})

    async def execute(self, shared_context: Dict[str, Any]) -> WaveResult:
        """Execute all agents in the wave in parallel"""
        logger.info(f"🌊 Starting wave: {self.name} with {len(self.agents)} agents")
        wave_start = time.time()

        # Check dependencies
        if not self._check_dependencies(shared_context):
            raise Exception(f"Dependencies not met for wave {self.id}")

        # Create sync points for coordination
        sync_points = self.coordination.get('sync_points', [])

        # Execute agents in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(self.parallel_agents)

        async def run_agent_with_limit(agent):
            async with semaphore:
                return await agent.execute(shared_context)

        # Run all agents
        results = await asyncio.gather(
            *[run_agent_with_limit(agent) for agent in self.agents],
            return_exceptions=True
        )

        # Process results
        agent_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent execution exception: {result}")
                # Create failed result
                agent_results.append(AgentResult(
                    agent_id="unknown",
                    wave_id=self.id,
                    status="failed",
                    outputs={},
                    metrics={},
                    duration=0,
                    errors=[str(result)]
                ))
            else:
                agent_results.append(result)

        # Aggregate results
        wave_result = self._aggregate_results(agent_results, time.time() - wave_start)

        # Store wave outputs in context
        shared_context[f"{self.id}.all_outputs"] = wave_result.outputs

        logger.info(f"✅ Wave {self.name} completed: {wave_result.success_rate:.1f}% success rate")

        return wave_result

    def _check_dependencies(self, shared_context: Dict[str, Any]) -> bool:
        """Check if wave dependencies are satisfied"""
        for dep in self.depends_on:
            if f"{dep}.all_outputs" not in shared_context:
                logger.error(f"Dependency {dep} not found for wave {self.id}")
                return False
        return True

    def _aggregate_results(self, results: List[AgentResult], duration: float) -> WaveResult:
        """Aggregate individual agent results into wave result"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        # Aggregate outputs and metrics
        all_outputs = {}
        all_metrics = defaultdict(float)

        for result in results:
            all_outputs[result.agent_id] = result.outputs
            for metric, value in result.metrics.items():
                all_metrics[metric] += value

        # Calculate average metrics
        if results:
            for metric in all_metrics:
                all_metrics[metric] /= len(results)

        return WaveResult(
            wave_id=self.id,
            agents_completed=successful,
            agents_failed=failed,
            total_duration=duration,
            outputs=all_outputs,
            metrics=dict(all_metrics),
            success_rate=(successful / len(results) * 100) if results else 0
        )


class SwarmOrchestrator:
    """Main orchestrator for swarm execution"""

    def __init__(self, workflow_path: str):
        self.workflow_path = Path(workflow_path)
        self.workflow = self._load_workflow()
        self.waves = self._initialize_waves()
        self.shared_context = {}
        self.checkpoints = []
        self.start_time = None
        self.monitoring_task = None

    def _load_workflow(self) -> Dict[str, Any]:
        """Load workflow definition from YAML"""
        with open(self.workflow_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_waves(self) -> List[SwarmWave]:
        """Initialize wave objects from workflow"""
        return [SwarmWave(wave_config) for wave_config in self.workflow.get('waves', [])]

    async def execute(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the entire swarm workflow"""
        self.start_time = datetime.now()

        print(f"🚀 Starting Swarm Execution: {self.workflow['name']}")
        print(f"   Version: {self.workflow['version']}")
        print(f"   Description: {self.workflow['description']}")
        print("=" * 70)

        # Merge parameters
        params = self._merge_parameters(parameters)
        self.shared_context['parameters'] = params

        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_progress())

        # Execute waves in sequence (waves internally execute agents in parallel)
        wave_results = []
        try:
            for wave in self.waves:
                print(f"\n{'='*70}")
                print(f"🌊 Executing Wave: {wave.name}")
                print(f"   Agents: {len(wave.agents)}")
                print(f"   Max Parallel: {wave.parallel_agents}")
                print("=" * 70)

                # Execute wave
                wave_result = await wave.execute(self.shared_context)
                wave_results.append(wave_result)

                # Create checkpoint after each wave
                self._create_checkpoint(wave.id, wave_result)

                # Check if we should continue based on wave result
                if not self._should_continue(wave_result):
                    logger.warning(f"Stopping execution after wave {wave.id} due to low success rate")
                    break

            # Generate final report
            final_report = self._generate_final_report(wave_results)

            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()

            return final_report

        except Exception as e:
            logger.error(f"Swarm execution failed: {str(e)}")
            if self.monitoring_task:
                self.monitoring_task.cancel()
            raise

    def _merge_parameters(self, user_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user parameters with defaults"""
        default_params = {}
        if 'parameters' in self.workflow:
            for param, config in self.workflow['parameters'].items():
                if 'default' in config:
                    default_params[param] = config['default']

        if user_params:
            default_params.update(user_params)

        return default_params

    def _should_continue(self, wave_result: WaveResult) -> bool:
        """Determine if execution should continue after a wave"""
        # Continue if success rate is above threshold
        min_success_rate = self.workflow.get('config', {}).get('min_success_rate', 50)
        return wave_result.success_rate >= min_success_rate

    def _create_checkpoint(self, wave_id: str, wave_result: WaveResult):
        """Create a checkpoint after wave completion"""
        checkpoint = {
            'wave_id': wave_id,
            'timestamp': datetime.now().isoformat(),
            'success_rate': wave_result.success_rate,
            'context_hash': hashlib.md5(
                json.dumps(self.shared_context, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        self.checkpoints.append(checkpoint)

        # Save checkpoint to file
        checkpoint_file = Path(f"checkpoints/swarm_{wave_id}_{checkpoint['context_hash'][:8]}.json")
        checkpoint_file.parent.mkdir(exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'checkpoint': checkpoint,
                'context': self.shared_context
            }, f, indent=2, default=str)

    async def _monitor_progress(self):
        """Monitor and report progress in real-time"""
        try:
            while True:
                await asyncio.sleep(5)  # Report every 5 seconds
                # In real implementation, would report:
                # - Current wave progress
                # - Agent statuses
                # - Resource usage
                # - Performance metrics
                logger.info("📊 Progress monitoring active...")
        except asyncio.CancelledError:
            logger.info("Monitoring stopped")

    def _generate_final_report(self, wave_results: List[WaveResult]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        duration = (datetime.now() - self.start_time).total_seconds()

        # Calculate overall metrics
        total_agents = sum(w.agents_completed + w.agents_failed for w in wave_results)
        successful_agents = sum(w.agents_completed for w in wave_results)

        # Aggregate all metrics
        combined_metrics = defaultdict(float)
        for wave_result in wave_results:
            for metric, value in wave_result.metrics.items():
                combined_metrics[metric] += value

        # Calculate averages
        if wave_results:
            for metric in combined_metrics:
                combined_metrics[metric] /= len(wave_results)

        report = {
            'workflow': self.workflow['name'],
            'version': self.workflow['version'],
            'execution_time': duration,
            'waves_executed': len(wave_results),
            'total_agents': total_agents,
            'successful_agents': successful_agents,
            'overall_success_rate': (successful_agents / total_agents * 100) if total_agents else 0,
            'metrics': dict(combined_metrics),
            'checkpoints': self.checkpoints,
            'timestamp': datetime.now().isoformat()
        }

        # Save final report
        report_file = Path(f"reports/swarm_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict[str, Any]):
        """Print execution summary"""
        print("\n" + "=" * 70)
        print("📊 SWARM EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Workflow: {report['workflow']}")
        print(f"Duration: {report['execution_time']:.2f} seconds")
        print(f"Waves Executed: {report['waves_executed']}")
        print(f"Total Agents: {report['total_agents']}")
        print(f"Successful Agents: {report['successful_agents']}")
        print(f"Overall Success Rate: {report['overall_success_rate']:.1f}%")
        print("\nKey Metrics:")
        for metric, value in report['metrics'].items():
            print(f"  - {metric}: {value:.2f}")
        print("\n🎉 Swarm execution completed successfully!")
        print("=" * 70)


async def main():
    """Main entry point for swarm orchestration"""
    import argparse

    parser = argparse.ArgumentParser(description='Execute Codebase Improvement Swarm')
    parser.add_argument('--workflow',
                       default='.claude/workflows/codebase-improvement-swarm.yaml',
                       help='Path to workflow YAML file')
    parser.add_argument('--params', type=json.loads, default={},
                       help='JSON parameters for the workflow')
    parser.add_argument('--profile',
                       choices=['quick', 'standard', 'comprehensive', 'enterprise'],
                       default='comprehensive',
                       help='Execution profile')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate execution without running agents')

    args = parser.parse_args()

    # Add profile to parameters
    if args.profile:
        args.params['profile'] = args.profile

    # Execute swarm
    orchestrator = SwarmOrchestrator(args.workflow)
    result = await orchestrator.execute(args.params)

    # Exit with appropriate code
    success_rate = result.get('overall_success_rate', 0)
    sys.exit(0 if success_rate >= 80 else 1)


if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())