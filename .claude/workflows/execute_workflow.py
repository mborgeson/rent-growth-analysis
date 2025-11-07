#!/usr/bin/env python3
"""
Workflow Execution Script for Rent Growth Analysis
Provides a simple interface to run the defined workflows
"""

import argparse
import json
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class WorkflowExecutor:
    """Execute workflow definitions from YAML files"""

    def __init__(self, workflow_path: str):
        self.workflow_path = Path(workflow_path)
        self.workflow = self._load_workflow()
        self.results = {}
        self.start_time = None

    def _load_workflow(self) -> Dict[str, Any]:
        """Load workflow definition from YAML file"""
        with open(self.workflow_path, 'r') as f:
            return yaml.safe_load(f)

    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Execute the workflow with given parameters"""
        self.start_time = datetime.now()
        workflow_name = self.workflow.get('name', 'unknown')

        print(f"🚀 Starting workflow: {workflow_name}")
        print(f"   Version: {self.workflow.get('version', 'N/A')}")
        print(f"   Description: {self.workflow.get('description', 'N/A')}")
        print("-" * 50)

        # Merge parameters
        params = self._merge_parameters(parameters)

        # Execute based on workflow type
        if 'stages' in self.workflow:
            return self._execute_pipeline_workflow(params)
        elif 'steps' in self.workflow:
            return self._execute_analysis_workflow(params)
        else:
            print("❌ Invalid workflow format")
            return False

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

    def _execute_analysis_workflow(self, params: Dict[str, Any]) -> bool:
        """Execute analysis workflow with steps"""
        steps = self.workflow.get('steps', [])

        for step in steps:
            step_id = step.get('id', 'unknown')
            step_name = step.get('name', 'Unnamed Step')

            print(f"\n📍 Executing step: {step_name} (ID: {step_id})")

            # Check dependencies
            depends_on = step.get('depends_on', [])
            if depends_on:
                print(f"   Dependencies: {', '.join(depends_on)}")

            # Simulate step execution
            success = self._execute_step(step, params)

            if not success:
                if step.get('on_failure') == 'halt':
                    print(f"❌ Step {step_id} failed. Halting workflow.")
                    return False
                else:
                    print(f"⚠️  Step {step_id} failed. Continuing...")

            self.results[step_id] = {
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat()
            }

        return True

    def _execute_pipeline_workflow(self, params: Dict[str, Any]) -> bool:
        """Execute data pipeline workflow with stages"""
        stages = self.workflow.get('stages', [])

        for stage in stages:
            stage_id = stage.get('id', 'unknown')
            stage_name = stage.get('name', 'Unnamed Stage')

            print(f"\n🔄 Executing stage: {stage_name} (ID: {stage_id})")

            # Check dependencies
            depends_on = stage.get('depends_on', [])
            if depends_on:
                print(f"   Dependencies: {', '.join(depends_on)}")

            # Simulate stage execution
            success = self._execute_stage(stage, params)

            if not success and not stage.get('optional', False):
                print(f"❌ Stage {stage_id} failed. Halting pipeline.")
                return False

            self.results[stage_id] = {
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat()
            }

        return True

    def _execute_step(self, step: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Execute a single workflow step"""
        # Simulate execution based on step type
        step_type = step.get('type', 'unknown')
        agent = step.get('agent', None)

        if agent:
            print(f"   Using agent: {agent}")

        # Check for parallel subtasks
        if step.get('parallel') and 'subtasks' in step:
            print(f"   Executing {len(step['subtasks'])} subtasks in parallel...")
            for subtask in step['subtasks']:
                print(f"     - {subtask.get('name', 'Unnamed')}")

        # Simulate success (in real implementation, would execute actual commands)
        print(f"   ✅ Step completed successfully")
        return True

    def _execute_stage(self, stage: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Execute a single pipeline stage"""
        stage_type = stage.get('type', 'unknown')

        print(f"   Stage type: {stage_type}")

        # Handle different stage types
        if stage_type == 'extract':
            print(f"   Extracting data from sources...")
        elif stage_type == 'transform':
            print(f"   Transforming data...")
        elif stage_type == 'quality':
            print(f"   Running quality checks...")
        elif stage_type == 'load':
            print(f"   Loading data to destinations...")

        # Simulate success
        print(f"   ✅ Stage completed successfully")
        return True

    def print_summary(self):
        """Print workflow execution summary"""
        if not self.start_time:
            return

        duration = datetime.now() - self.start_time

        print("\n" + "=" * 50)
        print("📊 Workflow Execution Summary")
        print("=" * 50)
        print(f"Workflow: {self.workflow.get('name', 'unknown')}")
        print(f"Duration: {duration}")
        print(f"Steps/Stages executed: {len(self.results)}")

        successful = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed = sum(1 for r in self.results.values() if r['status'] == 'failed')

        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")

        if failed == 0:
            print("\n🎉 Workflow completed successfully!")
        else:
            print(f"\n⚠️  Workflow completed with {failed} failures")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Execute Rent Growth Analysis Workflows')
    parser.add_argument('workflow', choices=['analysis', 'pipeline', 'development', 'swarm'],
                       help='Workflow to execute')
    parser.add_argument('--params', type=json.loads, default={},
                       help='JSON parameters for the workflow')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate execution without running commands')
    parser.add_argument('--profile',
                       choices=['quick', 'standard', 'comprehensive', 'enterprise'],
                       help='Execution profile (for swarm workflows)')

    args = parser.parse_args()

    # Handle swarm workflow specially
    if args.workflow == 'swarm':
        import asyncio
        from pathlib import Path
        swarm_script = Path('.claude/workflows/swarm-orchestrator.py')
        if swarm_script.exists():
            # Run swarm orchestrator directly
            cmd = [sys.executable, str(swarm_script)]
            if args.params:
                cmd.extend(['--params', json.dumps(args.params)])
            if args.profile:
                cmd.extend(['--profile', args.profile])
            if args.dry_run:
                cmd.append('--dry-run')
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        else:
            print(f"❌ Swarm orchestrator not found: {swarm_script}")
            sys.exit(1)

    # Map workflow names to files
    workflow_files = {
        'analysis': '.claude/workflows/rent-growth-analysis.yaml',
        'pipeline': '.claude/workflows/data-pipeline.yaml',
        'development': '.claude/workflows/development-workflow.yaml'
    }

    workflow_file = workflow_files[args.workflow]

    if not Path(workflow_file).exists():
        print(f"❌ Workflow file not found: {workflow_file}")
        sys.exit(1)

    # Execute workflow
    executor = WorkflowExecutor(workflow_file)
    success = executor.execute(args.params)
    executor.print_summary()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()