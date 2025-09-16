"""
Multi-Agent System Framework for Rent Growth Analysis

This package provides the base framework and specialized agents for the
hierarchical swarm architecture used in multifamily rent growth analysis.

Key Components:
- BaseAgent: Abstract base class for all agents
- AgentMessage: Inter-agent communication protocol
- AgentStatus: Agent state management
- ResourceMetrics: Resource monitoring and management
- ConsensusProposal: Consensus mechanism implementation

Author: Multi-Agent System Framework
Version: 1.0.0
"""

from .base_agent import (
    BaseAgent,
    AgentMessage,
    AgentStatus,
    MessageType,
    Priority,
    ResourceMetrics,
    ConsensusProposal
)

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentStatus",
    "MessageType",
    "Priority",
    "ResourceMetrics",
    "ConsensusProposal"
]