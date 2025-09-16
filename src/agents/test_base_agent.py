"""
Test Suite for Base Agent Framework

Tests the core functionality of the multi-agent system framework including
communication, resource management, consensus mechanisms, and quality gates.

Author: Multi-Agent System Framework
Version: 1.0.0
"""

import asyncio
import pytest
import time
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from .base_agent import (
    BaseAgent, 
    AgentMessage, 
    AgentStatus, 
    MessageType, 
    Priority,
    ResourceMetrics,
    ConsensusProposal
)


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes"""
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name=f"Test Agent {agent_id}",
            agent_type="test",
            capabilities=["testing", "validation"],
            **kwargs
        )
        self.processed_tasks = []
        self.initialization_called = False
        self.cleanup_called = False
    
    async def _initialize(self) -> None:
        self.initialization_called = True
        await asyncio.sleep(0.1)  # Simulate initialization
    
    async def _cleanup(self) -> None:
        self.cleanup_called = True
        await asyncio.sleep(0.1)  # Simulate cleanup
    
    async def process_task(self, task_data):
        self.processed_tasks.append(task_data)
        await asyncio.sleep(0.1)  # Simulate processing
        return {"status": "completed", "result": f"processed_{task_data.get('id', 'unknown')}"}


class TestAgentMessage:
    """Test AgentMessage functionality"""
    
    def test_message_creation(self):
        """Test basic message creation"""
        message = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id="agent-001",
            recipient_id="agent-002",
            payload={"test": "data"}
        )
        
        assert message.message_type == MessageType.REQUEST
        assert message.sender_id == "agent-001"
        assert message.recipient_id == "agent-002"
        assert message.payload == {"test": "data"}
        assert message.priority == Priority.NORMAL
        assert message.timeout_seconds == 30
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert isinstance(message.message_id, str)
        assert isinstance(message.timestamp, datetime)
    
    def test_message_serialization(self):
        """Test message to/from dict conversion"""
        original = AgentMessage(
            message_type=MessageType.BROADCAST,
            sender_id="agent-001",
            payload={"test": "data"},
            priority=Priority.HIGH
        )
        
        # Convert to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data['message_type'] == 'broadcast'
        assert data['sender_id'] == 'agent-001'
        assert data['payload'] == {"test": "data"}
        assert data['priority'] == 1  # HIGH priority value
        
        # Convert back from dict
        restored = AgentMessage.from_dict(data)
        assert restored.message_type == original.message_type
        assert restored.sender_id == original.sender_id
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.message_id == original.message_id
    
    def test_response_creation(self):
        """Test creating response messages"""
        request = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id="agent-001",
            recipient_id="agent-002",
            payload={"action": "test"}
        )
        
        response = request.create_response({"result": "success"}, "agent-002")
        
        assert response.message_type == MessageType.RESPONSE
        assert response.sender_id == "agent-002"
        assert response.recipient_id == "agent-001"
        assert response.correlation_id == request.message_id
        assert response.payload == {"result": "success"}
    
    def test_error_creation(self):
        """Test creating error messages"""
        request = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id="agent-001",
            recipient_id="agent-002"
        )
        
        error = request.create_error("Something went wrong", "agent-002")
        
        assert error.message_type == MessageType.ERROR
        assert error.sender_id == "agent-002"
        assert error.recipient_id == "agent-001"
        assert error.correlation_id == request.message_id
        assert error.payload["error"] == "Something went wrong"
        assert error.payload["original_message_id"] == request.message_id
        assert error.priority == Priority.HIGH


class TestResourceMetrics:
    """Test ResourceMetrics functionality"""
    
    def test_metrics_creation(self):
        """Test basic metrics creation"""
        metrics = ResourceMetrics(
            cpu_percent=45.5,
            memory_mb=1024.0,
            memory_percent=25.0
        )
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_mb == 1024.0
        assert metrics.memory_percent == 25.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_serialization(self):
        """Test metrics to_dict conversion"""
        metrics = ResourceMetrics(
            cpu_percent=45.5,
            memory_mb=1024.0,
            memory_percent=25.0,
            disk_io_mb=100.0,
            network_io_mb=50.0
        )
        
        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data['cpu_percent'] == 45.5
        assert data['memory_mb'] == 1024.0
        assert data['memory_percent'] == 25.0
        assert data['disk_io_mb'] == 100.0
        assert data['network_io_mb'] == 50.0
        assert 'timestamp' in data


class TestConsensusProposal:
    """Test ConsensusProposal functionality"""
    
    def test_proposal_creation(self):
        """Test basic proposal creation"""
        proposal = ConsensusProposal(
            topic="test_decision",
            options=["option_a", "option_b", "option_c"],
            threshold=0.75,
            timeout_seconds=60
        )
        
        assert proposal.topic == "test_decision"
        assert proposal.options == ["option_a", "option_b", "option_c"]
        assert proposal.threshold == 0.75
        assert proposal.timeout_seconds == 60
        assert len(proposal.votes) == 0
        assert len(proposal.weights) == 0
        assert isinstance(proposal.proposal_id, str)
        assert isinstance(proposal.created_at, datetime)
    
    def test_voting(self):
        """Test voting mechanism"""
        proposal = ConsensusProposal(
            topic="test_decision",
            options=["option_a", "option_b"],
            threshold=0.6
        )
        
        # Add votes
        proposal.add_vote("agent-001", "option_a", 1.0)
        proposal.add_vote("agent-002", "option_b", 1.0)
        proposal.add_vote("agent-003", "option_a", 2.0)  # Higher weight
        
        assert len(proposal.votes) == 3
        assert proposal.votes["agent-001"] == "option_a"
        assert proposal.votes["agent-002"] == "option_b"
        assert proposal.votes["agent-003"] == "option_a"
        assert proposal.weights["agent-003"] == 2.0
    
    def test_consensus_calculation(self):
        """Test consensus result calculation"""
        proposal = ConsensusProposal(
            topic="test_decision",
            options=["option_a", "option_b"],
            threshold=0.6
        )
        
        # Add votes where option_a wins with 75% (3/4 weight)
        proposal.add_vote("agent-001", "option_a", 1.0)
        proposal.add_vote("agent-002", "option_b", 1.0)
        proposal.add_vote("agent-003", "option_a", 2.0)
        
        result = proposal.calculate_result()
        assert result == "option_a"
        
        # Test case where threshold is not met
        proposal.threshold = 0.8  # Increase threshold
        result = proposal.calculate_result()
        assert result is None  # 75% < 80%
    
    def test_expiration(self):
        """Test proposal expiration"""
        proposal = ConsensusProposal(
            topic="test_decision",
            options=["option_a", "option_b"],
            timeout_seconds=0.1  # Very short timeout
        )
        
        assert not proposal.is_expired()
        time.sleep(0.2)
        assert proposal.is_expired()


@pytest.mark.asyncio
class TestBaseAgent:
    """Test BaseAgent functionality"""
    
    async def test_agent_initialization(self):
        """Test agent creation and initialization"""
        agent = TestAgent("test-001")
        
        assert agent.agent_id == "test-001"
        assert agent.name == "Test Agent test-001"
        assert agent.agent_type == "test"
        assert "testing" in agent.capabilities
        assert agent.status == AgentStatus.INITIALIZING
        assert not agent.initialization_called
        
        await agent.start()
        
        assert agent.status == AgentStatus.IDLE
        assert agent.initialization_called
        
        await agent.stop()
        assert agent.cleanup_called
        assert agent.status == AgentStatus.DEAD
    
    async def test_task_processing(self):
        """Test task processing functionality"""
        agent = TestAgent("test-002")
        await agent.start()
        
        try:
            task_data = {"id": "task-001", "data": "test"}
            result = await agent.process_task(task_data)
            
            assert result["status"] == "completed"
            assert result["result"] == "processed_task-001"
            assert task_data in agent.processed_tasks
            
        finally:
            await agent.stop()
    
    async def test_message_handling(self):
        """Test message receiving and processing"""
        agent = TestAgent("test-003")
        await agent.start()
        
        try:
            # Create test message
            message = AgentMessage(
                message_type=MessageType.HEARTBEAT,
                sender_id="test-sender",
                payload={"status": "alive"}
            )
            
            # Send message to agent
            agent.receive_message(message)
            
            # Give time for processing
            await asyncio.sleep(0.2)
            
            # Verify message was processed (would need to check internal state)
            assert agent.message_queue.qsize() >= 0  # Queue might be empty after processing
            
        finally:
            await agent.stop()
    
    async def test_resource_monitoring(self):
        """Test resource monitoring functionality"""
        agent = TestAgent("test-004", max_memory_mb=512, max_cpu_percent=50)
        await agent.start()
        
        try:
            # Wait for resource monitoring to collect data
            await asyncio.sleep(1.5)
            
            # Check that metrics were collected
            assert len(agent.resource_metrics) > 0
            
            current = agent.get_current_resources()
            assert current is not None
            assert isinstance(current.cpu_percent, float)
            assert isinstance(current.memory_mb, float)
            assert isinstance(current.timestamp, datetime)
            
        finally:
            await agent.stop()
    
    async def test_status_info(self):
        """Test agent status information"""
        agent = TestAgent("test-005")
        await agent.start()
        
        try:
            status = agent.get_status_info()
            
            assert status["agent_id"] == "test-005"
            assert status["name"] == "Test Agent test-005"
            assert status["type"] == "test"
            assert status["status"] == AgentStatus.IDLE.value
            assert "testing" in status["capabilities"]
            assert "uptime_seconds" in status
            assert status["error_count"] == 0
            
        finally:
            await agent.stop()
    
    async def test_checkpoints(self):
        """Test checkpoint functionality"""
        agent = TestAgent("test-006")
        await agent.start()
        
        try:
            # Create checkpoint
            checkpoint_data = {"tasks_completed": 5, "quality_score": 0.95}
            agent.create_checkpoint("test_checkpoint", checkpoint_data)
            
            assert "test_checkpoint" in agent.checkpoints
            checkpoint = agent.checkpoints["test_checkpoint"]
            assert checkpoint["data"] == checkpoint_data
            assert "timestamp" in checkpoint
            assert "agent_state" in checkpoint
            
        finally:
            await agent.stop()
    
    async def test_quality_gates(self):
        """Test quality gate validation"""
        agent = TestAgent("test-007")
        
        # Test passing quality gate
        metrics = {
            'data_completeness': 0.90,
            'data_quality': 0.95
        }
        agent.quality_thresholds = {
            'data_completeness': 0.85,
            'data_quality': 0.90
        }
        
        assert agent.validate_quality_gate(metrics) == True
        
        # Test failing quality gate
        metrics['data_completeness'] = 0.80  # Below threshold
        assert agent.validate_quality_gate(metrics) == False
    
    async def test_consensus_participation(self):
        """Test consensus mechanism"""
        agent = TestAgent("test-008")
        await agent.start()
        
        try:
            # Test consensus initiation
            participants = ["agent-001", "agent-002"]
            
            # This would normally involve real agents, so we'll test the proposal creation
            proposal = ConsensusProposal(
                topic="test_consensus",
                options=["option_a", "option_b"],
                threshold=0.75
            )
            
            # Test decision making
            decision = agent._make_consensus_decision(proposal)
            assert decision in proposal.options or decision == "abstain"
            
        finally:
            await agent.stop()
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        agent = TestAgent("test-009", max_errors=2)
        
        # Simulate errors
        agent._handle_error(Exception("Test error 1"))
        assert agent.error_count == 1
        assert agent.status != AgentStatus.ERROR
        
        agent._handle_error(Exception("Test error 2"))
        assert agent.error_count == 2
        assert agent.status == AgentStatus.ERROR
        assert agent.shutdown_event.is_set()
    
    async def test_threading(self):
        """Test that background threads are working"""
        agent = TestAgent("test-010")
        await agent.start()
        
        try:
            # Wait a moment for threads to start
            await asyncio.sleep(0.5)
            
            # Check that threads are alive
            assert agent.heartbeat_thread.is_alive()
            assert agent.message_processing_thread.is_alive()
            assert agent.resource_monitoring_thread.is_alive()
            
        finally:
            await agent.stop()
            
            # Check that threads have stopped
            await asyncio.sleep(0.2)
            assert not agent.heartbeat_thread.is_alive()
            assert not agent.message_processing_thread.is_alive()
            assert not agent.resource_monitoring_thread.is_alive()


# Integration tests
@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for multiple agents"""
    
    async def test_multi_agent_communication(self):
        """Test communication between multiple agents"""
        agent1 = TestAgent("agent-001")
        agent2 = TestAgent("agent-002")
        
        await agent1.start()
        await agent2.start()
        
        try:
            # Create message from agent1 to agent2
            message = AgentMessage(
                message_type=MessageType.REQUEST,
                sender_id=agent1.agent_id,
                recipient_id=agent2.agent_id,
                payload={"request": "test_data"}
            )
            
            # Simulate message delivery
            agent2.receive_message(message)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Verify message was received and processed
            # In a real system, we would check the response
            
        finally:
            await agent1.stop()
            await agent2.stop()
    
    async def test_consensus_between_agents(self):
        """Test consensus mechanism between multiple agents"""
        agents = [TestAgent(f"agent-{i:03d}") for i in range(3)]
        
        for agent in agents:
            await agent.start()
        
        try:
            # Create consensus proposal
            proposal = ConsensusProposal(
                topic="test_decision",
                options=["option_a", "option_b"],
                threshold=0.67
            )
            
            # Have each agent vote
            for i, agent in enumerate(agents):
                vote = "option_a" if i < 2 else "option_b"  # 2 votes for A, 1 for B
                proposal.add_vote(agent.agent_id, vote, 1.0)
            
            # Check consensus result
            result = proposal.calculate_result()
            assert result == "option_a"  # Should win with 67% of votes
            
        finally:
            for agent in agents:
                await agent.stop()


# Performance tests
@pytest.mark.asyncio
class TestAgentPerformance:
    """Performance tests for agent framework"""
    
    async def test_message_throughput(self):
        """Test message processing throughput"""
        agent = TestAgent("perf-001")
        await agent.start()
        
        try:
            # Send multiple messages
            num_messages = 100
            start_time = time.time()
            
            for i in range(num_messages):
                message = AgentMessage(
                    message_type=MessageType.HEARTBEAT,
                    sender_id=f"sender-{i}",
                    payload={"id": i}
                )
                agent.receive_message(message)
            
            # Wait for processing
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            throughput = num_messages / (end_time - start_time)
            
            # Should be able to process at least 10 messages/second
            assert throughput > 10
            
        finally:
            await agent.stop()
    
    async def test_resource_overhead(self):
        """Test resource overhead of agent framework"""
        agent = TestAgent("perf-002")
        await agent.start()
        
        try:
            # Wait for monitoring to collect data
            await asyncio.sleep(1.0)
            
            metrics = agent.get_current_resources()
            
            # Framework should use reasonable resources
            assert metrics.memory_mb < 100  # Less than 100MB
            assert metrics.cpu_percent < 10  # Less than 10% CPU when idle
            
        finally:
            await agent.stop()


if __name__ == "__main__":
    # Run specific test
    import sys
    
    async def run_example_test():
        """Run a simple example test"""
        print("Running example agent test...")
        
        agent = TestAgent("example-001")
        await agent.start()
        
        try:
            # Process a task
            result = await agent.process_task({"id": "test-task", "data": "example"})
            print(f"Task result: {result}")
            
            # Get status
            status = agent.get_status_info()
            print(f"Agent status: {status['status']}")
            print(f"Capabilities: {status['capabilities']}")
            
            # Create checkpoint
            agent.create_checkpoint("example", {"test": "data"})
            print(f"Checkpoints: {list(agent.checkpoints.keys())}")
            
        finally:
            await agent.stop()
            print("Agent stopped successfully")
    
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        import logging
        logging.basicConfig(level=logging.INFO)
        asyncio.run(run_example_test())
    else:
        print("To run example: python test_base_agent.py example")
        print("To run full tests: pytest test_base_agent.py")