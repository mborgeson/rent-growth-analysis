# Multi-Agent System Framework

This directory contains the base framework for the hierarchical swarm multi-agent system used in the rent growth analysis project.

## Overview

The multi-agent system is designed to handle complex time series analysis tasks through coordinated agents that specialize in different aspects of data processing, analysis, and validation. The system follows a hierarchical swarm architecture with consensus mechanisms and quality gates.

## Core Components

### BaseAgent (`base_agent.py`)

The abstract base class that provides core functionality for all agents:

- **Communication**: Message passing with request/response and broadcast patterns
- **Resource Management**: CPU and memory monitoring with configurable limits
- **Heartbeat Mechanism**: Automated health monitoring and status reporting
- **Error Handling**: Retry logic with exponential backoff and graceful degradation
- **Consensus Participation**: Weighted voting system for collaborative decision making
- **Quality Gates**: Checkpoint validation with configurable thresholds
- **Logging & Monitoring**: Comprehensive instrumentation and metrics collection

### Key Classes

#### `AgentMessage`
Inter-agent communication protocol with:
- Message types (request, response, broadcast, error, heartbeat, consensus)
- Routing information and correlation IDs
- Priority levels and timeout handling
- Retry mechanisms

#### `AgentStatus`
Agent state management:
- `IDLE`: Ready for work
- `WORKING`: Processing tasks
- `WAITING`: Waiting for dependencies
- `ERROR`: Error state requiring intervention
- `COMPLETED`: Task completed successfully
- `INITIALIZING`: Starting up
- `SHUTTING_DOWN`: Graceful shutdown in progress
- `DEAD`: Stopped/terminated

#### `ResourceMetrics`
System resource tracking:
- CPU usage percentage
- Memory usage (MB and percentage)
- Disk I/O metrics
- Network I/O metrics (when available)
- Timestamp for historical tracking

#### `ConsensusProposal`
Consensus mechanism implementation:
- Weighted voting system
- Configurable thresholds
- Timeout handling
- Result calculation and notification

## Agent Architecture

### Threaded Design

Each agent runs multiple background threads:

1. **Heartbeat Thread**: Sends periodic status updates
2. **Message Processing Thread**: Handles incoming messages
3. **Resource Monitoring Thread**: Tracks resource usage
4. **Main Thread**: Executes agent-specific tasks

### Communication Patterns

#### Request/Response
```python
# Send request
request = AgentMessage(
    message_type=MessageType.REQUEST,
    recipient_id="target-agent",
    payload={"task": "analyze_data", "data": data}
)
agent.send_message(request)

# Handle response in message handler
def handle_response(self, message: AgentMessage):
    result = message.payload.get("result")
    # Process result
```

#### Broadcast
```python
# Broadcast to all agents
broadcast = AgentMessage(
    message_type=MessageType.BROADCAST,
    recipient_id=None,  # None = broadcast
    payload={"announcement": "System maintenance in 5 minutes"}
)
agent.send_message(broadcast)
```

#### Consensus Voting
```python
# Initiate consensus
result = await agent.request_consensus(
    topic="model_selection",
    options=["random_forest", "xgboost", "lstm"],
    participants=["analyst-001", "analyst-002", "analyst-003"],
    threshold=0.75
)
```

### Quality Gates

Agents implement configurable quality gates for validation:

```python
# Define quality thresholds
agent.quality_thresholds = {
    'data_completeness': 0.85,
    'data_quality': 0.90,
    'model_performance': 0.85,
    'validation_score': 0.85
}

# Validate metrics
metrics = {
    'data_completeness': 0.92,
    'data_quality': 0.88
}
passed = agent.validate_quality_gate(metrics)
```

### Checkpoints

Create and validate checkpoints for state management:

```python
# Create checkpoint
agent.create_checkpoint("data_processed", {
    "records_processed": 10000,
    "quality_score": 0.95,
    "processing_time": 45.2
})

# Validate checkpoint
result = agent._validate_checkpoint("data_processed", {
    "min_records": 5000,
    "min_quality": 0.90
})
```

## Creating Custom Agents

### Basic Agent Implementation

```python
from src.agents import BaseAgent, AgentStatus

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="Custom Agent",
            agent_type="custom",
            capabilities=["custom_task", "data_processing"],
            **kwargs
        )
    
    async def _initialize(self) -> None:
        """Agent-specific initialization"""
        # Setup connections, load config, etc.
        pass
    
    async def _cleanup(self) -> None:
        """Agent-specific cleanup"""
        # Close connections, save state, etc.
        pass
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent-specific tasks"""
        self.status = AgentStatus.WORKING
        
        try:
            # Your processing logic here
            result = {"status": "completed", "data": task_data}
            
            self.status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise
        
        finally:
            self.status = AgentStatus.IDLE
```

### Advanced Features

#### Custom Message Handlers

```python
def setup_custom_handlers(self):
    self.message_handlers[MessageType.REQUEST] = self._handle_custom_request

def _handle_custom_request(self, message: AgentMessage):
    # Custom request handling logic
    if message.payload.get("action") == "custom_action":
        # Process custom action
        response = message.create_response({"result": "success"}, self.agent_id)
        self.send_message(response)
```

#### Custom Consensus Decision Making

```python
def _make_consensus_decision(self, proposal: ConsensusProposal) -> str:
    """Make intelligent decisions based on agent expertise"""
    if proposal.topic == "data_quality_threshold":
        # Agent prefers higher quality
        quality_options = [opt for opt in proposal.options if "high" in opt]
        return quality_options[0] if quality_options else proposal.options[0]
    
    return super()._make_consensus_decision(proposal)
```

#### Custom Quality Validation

```python
def _validate_checkpoint(self, checkpoint_id: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Custom validation logic"""
    scores = {}
    passed = True
    details = []
    
    # Your custom validation logic
    if criteria.get("custom_metric"):
        # Validate custom metric
        pass
    
    return {
        'passed': passed,
        'scores': scores,
        'details': details
    }
```

## Resource Management

### Memory and CPU Limits

```python
agent = CustomAgent(
    agent_id="agent-001",
    max_memory_mb=2048,  # 2GB limit
    max_cpu_percent=75.0  # 75% CPU limit
)
```

### Resource Monitoring

```python
# Get current resource usage
current = agent.get_current_resources()
print(f"Memory: {current.memory_mb}MB ({current.memory_percent}%)")
print(f"CPU: {current.cpu_percent}%")

# Get resource history
history = agent.resource_metrics[-10:]  # Last 10 measurements
```

## Example Usage

See `example_agent.py` for a complete implementation example that demonstrates:

- Task processing with validation
- Error handling and recovery
- Resource monitoring
- Quality gate validation
- Consensus participation
- Statistics tracking

## Integration with System

The base agent framework integrates with the broader system components:

- **Message Broker**: RabbitMQ for inter-agent communication
- **Database**: PostgreSQL for state persistence
- **Monitoring**: Elasticsearch for log aggregation
- **Orchestration**: Master coordinator for task distribution
- **Quality Control**: Automated validation and quality gates

## Configuration

Agents are configured through the main `agent_orchestration_config.yaml` file, which defines:

- Agent specifications and capabilities
- Resource limits and allocation
- Communication patterns and timeouts
- Consensus mechanisms and voting weights
- Quality thresholds and validation criteria
- Monitoring and alerting rules

## Performance Considerations

- **Threading**: Each agent uses multiple threads for different concerns
- **Resource Limits**: Configurable CPU and memory limits prevent resource exhaustion
- **Message Queuing**: Asynchronous message processing with queuing
- **Batch Processing**: Support for batched operations to improve efficiency
- **Caching**: Result caching to avoid redundant computations
- **Monitoring**: Continuous resource monitoring with alerting

## Security

- **Message Validation**: All messages are validated before processing
- **Resource Isolation**: Agents operate within defined resource boundaries
- **Error Isolation**: Agent failures don't cascade to other agents
- **Authentication**: Integration with JWT-based authentication system
- **Audit Logging**: Comprehensive audit trails for all agent actions

## Future Enhancements

- **Dynamic Load Balancing**: Automatic load distribution based on agent performance
- **Machine Learning Integration**: Intelligent task routing and resource allocation
- **Distributed Consensus**: Support for distributed consensus algorithms
- **Hot Swapping**: Runtime agent replacement without system downtime
- **Advanced Monitoring**: Real-time performance dashboards and predictive analytics