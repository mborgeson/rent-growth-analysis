"""
Base Agent Framework for Multi-Agent System
Hierarchical Swarm with Consensus Mechanisms

This module provides the foundational classes and interfaces for the multi-agent
rent growth analysis system, including communication, resource management,
consensus mechanisms, and quality gates.

Author: Multi-Agent System Framework
Version: 1.0.0
"""

import abc
import asyncio
import json
import logging
import psutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Set
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import traceback


class AgentStatus(Enum):
    """Agent operational states"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"
    DEAD = "dead"


class MessageType(Enum):
    """Inter-agent message types"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_RESULT = "consensus_result"
    CHECKPOINT = "checkpoint"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_GRANT = "resource_grant"
    SHUTDOWN = "shutdown"


class Priority(Enum):
    """Task and message priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ResourceMetrics:
    """System resource usage metrics"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'disk_io_mb': self.disk_io_mb,
            'network_io_mb': self.network_io_mb,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: Priority = Priority.NORMAL
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3
    routing_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'correlation_id': self.correlation_id,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'routing_key': self.routing_key
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=MessageType(data.get('message_type', 'request')),
            sender_id=data.get('sender_id', ''),
            recipient_id=data.get('recipient_id'),
            correlation_id=data.get('correlation_id'),
            payload=data.get('payload', {}),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now(timezone.utc).isoformat())),
            priority=Priority(data.get('priority', 2)),
            timeout_seconds=data.get('timeout_seconds', 30),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            routing_key=data.get('routing_key')
        )
    
    def create_response(self, payload: Dict[str, Any], sender_id: str) -> 'AgentMessage':
        """Create a response message to this message"""
        return AgentMessage(
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            correlation_id=self.message_id,
            payload=payload,
            priority=self.priority
        )
    
    def create_error(self, error_message: str, sender_id: str) -> 'AgentMessage':
        """Create an error response to this message"""
        return AgentMessage(
            message_type=MessageType.ERROR,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            correlation_id=self.message_id,
            payload={
                'error': error_message,
                'original_message_id': self.message_id
            },
            priority=Priority.HIGH
        )


@dataclass
class ConsensusProposal:
    """Consensus voting proposal"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    options: List[str] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)  # agent_id -> vote
    weights: Dict[str, float] = field(default_factory=dict)  # agent_id -> weight
    threshold: float = 0.75
    timeout_seconds: int = 60
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_vote(self, agent_id: str, vote: str, weight: float = 1.0) -> None:
        """Add a vote to the proposal"""
        self.votes[agent_id] = vote
        self.weights[agent_id] = weight
    
    def calculate_result(self) -> Optional[str]:
        """Calculate consensus result based on weighted votes"""
        if not self.votes:
            return None
        
        vote_counts = {}
        total_weight = sum(self.weights.values())
        
        for agent_id, vote in self.votes.items():
            weight = self.weights.get(agent_id, 1.0)
            vote_counts[vote] = vote_counts.get(vote, 0) + weight
        
        # Find option with highest weighted vote
        winner = max(vote_counts.items(), key=lambda x: x[1])
        winner_option, winner_weight = winner
        
        # Check if it meets threshold
        if winner_weight / total_weight >= self.threshold:
            return winner_option
        
        return None
    
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        elapsed = datetime.now(timezone.utc) - self.created_at
        return elapsed.total_seconds() > self.timeout_seconds


class BaseAgent(abc.ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Provides core functionality for:
    - Communication and message passing
    - Resource monitoring and management
    - Heartbeat mechanism
    - Error handling and recovery
    - Consensus participation
    - Quality gates and checkpoints
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None,
        max_memory_mb: float = 1024,
        max_cpu_percent: float = 80.0,
        heartbeat_interval: int = 30
    ):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.config = config or {}
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.heartbeat_interval = heartbeat_interval
        
        # State management
        self.status = AgentStatus.INITIALIZING
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = None
        self.error_count = 0
        self.max_errors = 5
        
        # Communication
        self.message_queue: Queue = Queue()
        self.response_futures: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.broadcast_subscribers: Set[str] = set()
        
        # Consensus
        self.consensus_proposals: Dict[str, ConsensusProposal] = {}
        self.voting_weights: Dict[str, float] = {}
        
        # Resource monitoring
        self.resource_metrics: List[ResourceMetrics] = []
        self.max_metrics_history = 1000
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.shutdown_event = threading.Event()
        self.heartbeat_thread = None
        self.message_processing_thread = None
        self.resource_monitoring_thread = None
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.agent_id}")
        
        # Quality gates
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.quality_thresholds = {
            'data_completeness': 0.85,
            'data_quality': 0.90,
            'model_performance': 0.85,
            'validation_score': 0.85
        }
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        self.logger.info(f"Agent {self.agent_id} ({self.name}) initialized with capabilities: {self.capabilities}")
    
    def _setup_message_handlers(self) -> None:
        """Setup default message handlers"""
        self.message_handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.SHUTDOWN: self._handle_shutdown,
            MessageType.CONSENSUS_VOTE: self._handle_consensus_vote,
            MessageType.CONSENSUS_RESULT: self._handle_consensus_result,
            MessageType.RESOURCE_REQUEST: self._handle_resource_request,
            MessageType.CHECKPOINT: self._handle_checkpoint
        }
    
    async def start(self) -> None:
        """Start the agent and all background threads"""
        try:
            self.logger.info(f"Starting agent {self.agent_id}")
            self.status = AgentStatus.IDLE
            
            # Start background threads
            self._start_heartbeat_thread()
            self._start_message_processing_thread()
            self._start_resource_monitoring_thread()
            
            # Agent-specific initialization
            await self._initialize()
            
            self.logger.info(f"Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources"""
        try:
            self.logger.info(f"Stopping agent {self.agent_id}")
            self.status = AgentStatus.SHUTTING_DOWN
            
            # Signal shutdown to all threads
            self.shutdown_event.set()
            
            # Wait for threads to finish
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5)
            
            if self.message_processing_thread and self.message_processing_thread.is_alive():
                self.message_processing_thread.join(timeout=5)
            
            if self.resource_monitoring_thread and self.resource_monitoring_thread.is_alive():
                self.resource_monitoring_thread.join(timeout=5)
            
            # Cleanup
            self.executor.shutdown(wait=True)
            await self._cleanup()
            
            self.status = AgentStatus.DEAD
            self.logger.info(f"Agent {self.agent_id} stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {self.agent_id}: {e}")
    
    @abc.abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization logic"""
        pass
    
    @abc.abstractmethod
    async def _cleanup(self) -> None:
        """Agent-specific cleanup logic"""
        pass
    
    @abc.abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific task - implemented by subclasses"""
        pass
    
    def _start_heartbeat_thread(self) -> None:
        """Start the heartbeat thread"""
        def heartbeat_loop():
            while not self.shutdown_event.is_set():
                try:
                    self._send_heartbeat()
                    self.last_heartbeat = datetime.now(timezone.utc)
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    time.sleep(1)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def _start_message_processing_thread(self) -> None:
        """Start the message processing thread"""
        def message_loop():
            while not self.shutdown_event.is_set():
                try:
                    try:
                        message = self.message_queue.get(timeout=1)
                        self._process_message(message)
                    except Empty:
                        continue
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    self._handle_error(e)
        
        self.message_processing_thread = threading.Thread(target=message_loop, daemon=True)
        self.message_processing_thread.start()
    
    def _start_resource_monitoring_thread(self) -> None:
        """Start the resource monitoring thread"""
        def monitor_loop():
            while not self.shutdown_event.is_set():
                try:
                    metrics = self._collect_resource_metrics()
                    self.resource_metrics.append(metrics)
                    
                    # Keep only recent metrics
                    if len(self.resource_metrics) > self.max_metrics_history:
                        self.resource_metrics = self.resource_metrics[-self.max_metrics_history:]
                    
                    # Check resource limits
                    if metrics.memory_mb > self.max_memory_mb:
                        self.logger.warning(f"Memory usage ({metrics.memory_mb}MB) exceeds limit ({self.max_memory_mb}MB)")
                    
                    if metrics.cpu_percent > self.max_cpu_percent:
                        self.logger.warning(f"CPU usage ({metrics.cpu_percent}%) exceeds limit ({self.max_cpu_percent}%)")
                    
                    time.sleep(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
        
        self.resource_monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.resource_monitoring_thread.start()
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource usage metrics"""
        process = psutil.Process()
        
        return ResourceMetrics(
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / (1024 * 1024),
            memory_percent=process.memory_percent(),
            disk_io_mb=sum(process.io_counters()[:2]) / (1024 * 1024),
            network_io_mb=0.0  # Would need additional libraries for network metrics
        )
    
    def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent or broadcast"""
        message.sender_id = self.agent_id
        self.logger.debug(f"Sending message {message.message_id} to {message.recipient_id or 'broadcast'}")
        
        # In a real implementation, this would send via message broker
        # For now, we'll simulate by adding to a shared queue or calling recipient directly
        self._route_message(message)
    
    def _route_message(self, message: AgentMessage) -> None:
        """Route message to appropriate destination"""
        # This is a simplified routing mechanism
        # In production, this would integrate with RabbitMQ or similar
        pass
    
    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent"""
        self.logger.debug(f"Received message {message.message_id} from {message.sender_id}")
        self.message_queue.put(message)
    
    def _process_message(self, message: AgentMessage) -> None:
        """Process an incoming message"""
        try:
            # Handle response correlation
            if message.message_type == MessageType.RESPONSE and message.correlation_id:
                future = self.response_futures.get(message.correlation_id)
                if future and not future.done():
                    future.set_result(message)
                    return
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                handler(message)
            else:
                self.logger.warning(f"No handler for message type {message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            if message.message_type != MessageType.ERROR:
                error_response = message.create_error(str(e), self.agent_id)
                self.send_message(error_response)
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat message"""
        heartbeat = AgentMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            payload={
                'status': self.status.value,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'resource_usage': self.get_current_resources().to_dict() if self.resource_metrics else None
            }
        )
        self.send_message(heartbeat)
    
    def _handle_heartbeat(self, message: AgentMessage) -> None:
        """Handle incoming heartbeat message"""
        self.logger.debug(f"Received heartbeat from {message.sender_id}")
        # Update agent registry or health monitoring
    
    def _handle_shutdown(self, message: AgentMessage) -> None:
        """Handle shutdown message"""
        self.logger.info(f"Received shutdown message from {message.sender_id}")
        self.shutdown_event.set()
    
    def _handle_consensus_vote(self, message: AgentMessage) -> None:
        """Handle consensus voting request"""
        proposal_id = message.payload.get('proposal_id')
        if proposal_id not in self.consensus_proposals:
            self.logger.warning(f"Received vote request for unknown proposal {proposal_id}")
            return
        
        proposal = self.consensus_proposals[proposal_id]
        vote = self._make_consensus_decision(proposal)
        weight = self.voting_weights.get(proposal.topic, 1.0)
        
        proposal.add_vote(self.agent_id, vote, weight)
        
        # Send vote response
        response = message.create_response({
            'proposal_id': proposal_id,
            'vote': vote,
            'weight': weight
        }, self.agent_id)
        self.send_message(response)
    
    def _handle_consensus_result(self, message: AgentMessage) -> None:
        """Handle consensus result notification"""
        proposal_id = message.payload.get('proposal_id')
        result = message.payload.get('result')
        self.logger.info(f"Consensus result for {proposal_id}: {result}")
    
    def _handle_resource_request(self, message: AgentMessage) -> None:
        """Handle resource allocation request"""
        requested_memory = message.payload.get('memory_mb', 0)
        requested_cpu = message.payload.get('cpu_percent', 0)
        
        current = self.get_current_resources()
        available_memory = self.max_memory_mb - current.memory_mb
        available_cpu = self.max_cpu_percent - current.cpu_percent
        
        can_grant = (requested_memory <= available_memory and 
                    requested_cpu <= available_cpu)
        
        response = message.create_response({
            'granted': can_grant,
            'available_memory_mb': available_memory,
            'available_cpu_percent': available_cpu
        }, self.agent_id)
        self.send_message(response)
    
    def _handle_checkpoint(self, message: AgentMessage) -> None:
        """Handle checkpoint validation request"""
        checkpoint_id = message.payload.get('checkpoint_id')
        criteria = message.payload.get('criteria', {})
        
        result = self._validate_checkpoint(checkpoint_id, criteria)
        
        response = message.create_response({
            'checkpoint_id': checkpoint_id,
            'passed': result['passed'],
            'scores': result['scores'],
            'details': result['details']
        }, self.agent_id)
        self.send_message(response)
    
    def _make_consensus_decision(self, proposal: ConsensusProposal) -> str:
        """Make a decision for consensus voting - override in subclasses"""
        # Default implementation returns first option
        return proposal.options[0] if proposal.options else "abstain"
    
    def _validate_checkpoint(self, checkpoint_id: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate checkpoint criteria - override in subclasses"""
        # Default implementation
        return {
            'passed': True,
            'scores': {},
            'details': f"Checkpoint {checkpoint_id} validation not implemented"
        }
    
    def _handle_error(self, error: Exception) -> None:
        """Handle errors with retry logic"""
        self.error_count += 1
        self.logger.error(f"Agent error ({self.error_count}/{self.max_errors}): {error}")
        
        if self.error_count >= self.max_errors:
            self.logger.critical(f"Agent {self.agent_id} exceeded maximum errors, shutting down")
            self.status = AgentStatus.ERROR
            self.shutdown_event.set()
    
    def get_current_resources(self) -> Optional[ResourceMetrics]:
        """Get current resource usage"""
        return self.resource_metrics[-1] if self.resource_metrics else None
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive agent status information"""
        current_resources = self.get_current_resources()
        uptime = datetime.now(timezone.utc) - self.start_time
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'type': self.agent_type,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'uptime_seconds': uptime.total_seconds(),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'error_count': self.error_count,
            'current_resources': current_resources.to_dict() if current_resources else None,
            'message_queue_size': self.message_queue.qsize(),
            'active_consensus_proposals': len(self.consensus_proposals),
            'checkpoints': list(self.checkpoints.keys())
        }
    
    def create_checkpoint(self, checkpoint_id: str, data: Dict[str, Any]) -> None:
        """Create a checkpoint with current state"""
        self.checkpoints[checkpoint_id] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': data,
            'agent_state': self.get_status_info()
        }
        self.logger.info(f"Created checkpoint {checkpoint_id}")
    
    def validate_quality_gate(self, metrics: Dict[str, float]) -> bool:
        """Validate quality gate thresholds"""
        passed = True
        for metric, value in metrics.items():
            threshold = self.quality_thresholds.get(metric, 0.0)
            if value < threshold:
                self.logger.warning(f"Quality gate failed: {metric} = {value} < {threshold}")
                passed = False
        
        return passed
    
    async def request_consensus(self, topic: str, options: List[str], 
                              participants: List[str], threshold: float = 0.75,
                              timeout: int = 60) -> Optional[str]:
        """Initiate a consensus voting process"""
        proposal = ConsensusProposal(
            topic=topic,
            options=options,
            threshold=threshold,
            timeout_seconds=timeout
        )
        
        self.consensus_proposals[proposal.proposal_id] = proposal
        
        # Send voting requests to all participants
        for participant_id in participants:
            vote_request = AgentMessage(
                message_type=MessageType.CONSENSUS_VOTE,
                sender_id=self.agent_id,
                recipient_id=participant_id,
                payload={
                    'proposal_id': proposal.proposal_id,
                    'topic': topic,
                    'options': options,
                    'threshold': threshold
                },
                timeout_seconds=timeout
            )
            self.send_message(vote_request)
        
        # Wait for votes or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(proposal.votes) >= len(participants):
                break
            await asyncio.sleep(1)
        
        # Calculate result
        result = proposal.calculate_result()
        
        # Notify all participants of result
        for participant_id in participants:
            result_message = AgentMessage(
                message_type=MessageType.CONSENSUS_RESULT,
                sender_id=self.agent_id,
                recipient_id=participant_id,
                payload={
                    'proposal_id': proposal.proposal_id,
                    'result': result,
                    'votes': proposal.votes
                }
            )
            self.send_message(result_message)
        
        # Cleanup
        del self.consensus_proposals[proposal.proposal_id]
        
        return result
    
    def __repr__(self) -> str:
        return f"BaseAgent(id={self.agent_id}, name={self.name}, type={self.agent_type}, status={self.status.value})"