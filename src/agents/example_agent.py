"""
Example Agent Implementation

Demonstrates how to create concrete agents using the BaseAgent framework.
This example shows a simple data processing agent that can be used as a
template for creating specialized agents.

Author: Multi-Agent System Framework
Version: 1.0.0
"""

import asyncio
import logging
from typing import Any, Dict, List
from datetime import datetime, timezone

from .base_agent import BaseAgent, AgentStatus, ConsensusProposal


class ExampleDataProcessor(BaseAgent):
    """
    Example implementation of a data processing agent.
    
    This agent demonstrates:
    - Task processing with error handling
    - Resource management
    - Quality gate validation
    - Consensus participation
    """
    
    def __init__(self, agent_id: str, name: str = "Example Data Processor", **kwargs):
        super().__init__(
            agent_id=agent_id,
            name=name,
            agent_type="data_processor",
            capabilities=[
                "data_validation",
                "data_transformation", 
                "quality_assessment",
                "consensus_participation"
            ],
            **kwargs
        )
        
        # Agent-specific configuration
        self.processing_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
        self.processing_stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    async def _initialize(self) -> None:
        """Initialize the data processor agent"""
        self.logger.info(f"Initializing {self.name}")
        
        # Agent-specific initialization
        await self._load_configuration()
        await self._setup_data_connections()
        
        self.logger.info(f"{self.name} initialization complete")
    
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info(f"Cleaning up {self.name}")
        
        # Save any pending work
        if self.processing_queue:
            self.logger.warning(f"Shutting down with {len(self.processing_queue)} pending tasks")
        
        # Close connections, save state, etc.
        await self._save_state()
        
        self.logger.info(f"{self.name} cleanup complete")
    
    async def _load_configuration(self) -> None:
        """Load agent-specific configuration"""
        # In a real implementation, this would load from config files, database, etc.
        self.logger.debug("Loading configuration")
        
        # Set quality thresholds
        self.quality_thresholds.update({
            'data_completeness': 0.90,
            'data_quality': 0.85,
            'processing_accuracy': 0.95
        })
    
    async def _setup_data_connections(self) -> None:
        """Setup data source connections"""
        # In a real implementation, this would setup database connections,
        # API clients, file system access, etc.
        self.logger.debug("Setting up data connections")
        await asyncio.sleep(0.1)  # Simulate connection setup
    
    async def _save_state(self) -> None:
        """Save current agent state"""
        state = {
            'processing_stats': self.processing_stats,
            'completed_tasks': len(self.completed_tasks),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # In a real implementation, this would persist to storage
        self.logger.debug(f"Saving agent state: {state}")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a data processing task
        
        Args:
            task_data: Dictionary containing task parameters
            
        Returns:
            Dictionary containing processing results
        """
        task_id = task_data.get('task_id', 'unknown')
        start_time = datetime.now(timezone.utc)
        
        try:
            self.status = AgentStatus.WORKING
            self.logger.info(f"Processing task {task_id}")
            
            # Validate input data
            validation_result = await self._validate_input(task_data)
            if not validation_result['valid']:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")
            
            # Process the data
            result = await self._process_data(task_data)
            
            # Validate output
            if not await self._validate_output(result):
                raise ValueError("Output validation failed")
            
            # Update statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_stats(processing_time, success=True)
            
            # Create checkpoint
            self.create_checkpoint(f"task_{task_id}", {
                'task_data': task_data,
                'result': result,
                'processing_time': processing_time
            })
            
            self.status = AgentStatus.COMPLETED
            self.logger.info(f"Successfully processed task {task_id} in {processing_time:.2f}s")
            
            return {
                'task_id': task_id,
                'status': 'completed',
                'result': result,
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_stats(processing_time, success=False)
            
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to process task {task_id}: {e}")
            
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        finally:
            if self.status not in [AgentStatus.ERROR, AgentStatus.SHUTTING_DOWN]:
                self.status = AgentStatus.IDLE
    
    async def _validate_input(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        errors = []
        
        # Check required fields
        required_fields = ['task_id', 'data']
        for field in required_fields:
            if field not in task_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate data structure
        if 'data' in task_data:
            data = task_data['data']
            if not isinstance(data, (list, dict)):
                errors.append("Data must be a list or dictionary")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _process_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the actual data"""
        data = task_data['data']
        
        # Simulate data processing
        await asyncio.sleep(0.5)  # Simulate processing time
        
        if isinstance(data, list):
            # Process list data
            processed = [self._transform_item(item) for item in data]
            return {
                'type': 'list',
                'count': len(processed),
                'processed_data': processed,
                'quality_score': 0.95
            }
        
        elif isinstance(data, dict):
            # Process dictionary data
            processed = {k: self._transform_item(v) for k, v in data.items()}
            return {
                'type': 'dict',
                'count': len(processed),
                'processed_data': processed,
                'quality_score': 0.92
            }
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _transform_item(self, item: Any) -> Any:
        """Transform a single data item"""
        # Simple transformation logic
        if isinstance(item, str):
            return item.upper().strip()
        elif isinstance(item, (int, float)):
            return item * 1.1  # Apply some factor
        else:
            return str(item)
    
    async def _validate_output(self, result: Dict[str, Any]) -> bool:
        """Validate processing output"""
        # Check quality score
        quality_score = result.get('quality_score', 0.0)
        if quality_score < self.quality_thresholds.get('processing_accuracy', 0.95):
            self.logger.warning(f"Quality score {quality_score} below threshold")
            return False
        
        # Check data completeness
        if 'processed_data' not in result:
            self.logger.error("Missing processed_data in result")
            return False
        
        return True
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics"""
        if success:
            self.processing_stats['tasks_completed'] += 1
        else:
            self.processing_stats['tasks_failed'] += 1
        
        self.processing_stats['total_processing_time'] += processing_time
        
        total_tasks = (self.processing_stats['tasks_completed'] + 
                      self.processing_stats['tasks_failed'])
        
        if total_tasks > 0:
            self.processing_stats['average_processing_time'] = (
                self.processing_stats['total_processing_time'] / total_tasks
            )
    
    def _make_consensus_decision(self, proposal: ConsensusProposal) -> str:
        """Make intelligent consensus decisions based on agent expertise"""
        topic = proposal.topic.lower()
        
        # Decision logic based on agent's data processing expertise
        if 'data_quality' in topic:
            # Prefer higher quality thresholds
            quality_options = [opt for opt in proposal.options if 'high' in opt.lower()]
            return quality_options[0] if quality_options else proposal.options[0]
        
        elif 'processing_method' in topic:
            # Prefer batch processing for efficiency
            batch_options = [opt for opt in proposal.options if 'batch' in opt.lower()]
            return batch_options[0] if batch_options else proposal.options[0]
        
        else:
            # Default to first option
            return proposal.options[0] if proposal.options else "abstain"
    
    def _validate_checkpoint(self, checkpoint_id: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate checkpoint criteria"""
        scores = {}
        passed = True
        details = []
        
        # Validate task completion rate
        total_tasks = (self.processing_stats['tasks_completed'] + 
                      self.processing_stats['tasks_failed'])
        
        if total_tasks > 0:
            completion_rate = self.processing_stats['tasks_completed'] / total_tasks
            scores['completion_rate'] = completion_rate
            
            required_rate = criteria.get('completion_rate', 0.90)
            if completion_rate < required_rate:
                passed = False
                details.append(f"Completion rate {completion_rate:.2f} below required {required_rate}")
        
        # Validate average processing time
        avg_time = self.processing_stats['average_processing_time']
        scores['average_processing_time'] = avg_time
        
        max_time = criteria.get('max_processing_time', 10.0)
        if avg_time > max_time:
            passed = False
            details.append(f"Average processing time {avg_time:.2f}s exceeds limit {max_time}s")
        
        # Validate resource usage
        current_resources = self.get_current_resources()
        if current_resources:
            scores['memory_usage'] = current_resources.memory_percent
            scores['cpu_usage'] = current_resources.cpu_percent
            
            max_memory = criteria.get('max_memory_percent', 80.0)
            max_cpu = criteria.get('max_cpu_percent', 70.0)
            
            if current_resources.memory_percent > max_memory:
                passed = False
                details.append(f"Memory usage {current_resources.memory_percent:.1f}% exceeds limit {max_memory}%")
            
            if current_resources.cpu_percent > max_cpu:
                passed = False
                details.append(f"CPU usage {current_resources.cpu_percent:.1f}% exceeds limit {max_cpu}%")
        
        return {
            'passed': passed,
            'scores': scores,
            'details': details
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()


# Example usage and testing
async def example_usage():
    """Example of how to use the agent"""
    # Create agent
    agent = ExampleDataProcessor("example-001")
    
    try:
        # Start agent
        await agent.start()
        
        # Process some tasks
        tasks = [
            {
                'task_id': 'task-001',
                'data': ['hello', 'world', 'test']
            },
            {
                'task_id': 'task-002', 
                'data': {'key1': 'value1', 'key2': 42}
            }
        ]
        
        for task in tasks:
            result = await agent.process_task(task)
            print(f"Task result: {result}")
        
        # Get status
        status = agent.get_status_info()
        print(f"Agent status: {status}")
        
        # Get processing stats
        stats = agent.get_processing_stats()
        print(f"Processing stats: {stats}")
        
    finally:
        # Stop agent
        await agent.stop()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())