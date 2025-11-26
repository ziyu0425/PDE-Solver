"""
Conversation Memory Module
--------------------------
Stores conversation history and previous simulation parameters for context-aware follow-up questions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import json
import os


@dataclass
class SimulationRecord:
    """Record of a previous simulation."""
    timestamp: datetime
    user_query: str
    pde_params: Optional[Dict[str, Any]] = None
    solver_result: Optional[Dict[str, Any]] = None
    html_path: Optional[str] = None
    data_file: Optional[str] = None
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_query": self.user_query,
            "pde_params": self.pde_params,
            "solver_result": self.solver_result,
            "html_path": self.html_path,
            "data_file": self.data_file,
            "summary": self.summary,
        }


class ConversationMemory:
    """Manages conversation history and simulation records."""
    
    def __init__(self, max_messages: int = 50, persist_file: Optional[str] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory.
            persist_file: Optional file path to persist/load conversation history.
        """
        self.max_messages = max_messages
        self.persist_file = persist_file
        self.messages: List[BaseMessage] = []
        self.simulation_history: List[SimulationRecord] = []
        self.current_simulation: Optional[SimulationRecord] = None
        
        # Load persisted memory if file exists and is not a directory
        if persist_file and os.path.exists(persist_file) and os.path.isfile(persist_file):
            self.load()
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.messages.append(message)
        
        # Trim to max_messages (keep oldest messages)
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent messages
            system_msgs = [msg for msg in self.messages if isinstance(msg, SystemMessage)]
            non_system_msgs = [msg for msg in self.messages if not isinstance(msg, SystemMessage)]
            
            # Keep last (max_messages - len(system_msgs)) non-system messages
            keep_count = self.max_messages - len(system_msgs)
            if len(non_system_msgs) > keep_count:
                non_system_msgs = non_system_msgs[-keep_count:]
            
            self.messages = system_msgs + non_system_msgs
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add an AI/agent message."""
        self.add_message(AIMessage(content=content))
    
    def record_simulation(
        self,
        user_query: str,
        pde_params: Optional[Dict[str, Any]] = None,
        solver_result: Optional[Dict[str, Any]] = None,
        html_path: Optional[str] = None,
        data_file: Optional[str] = None,
        summary: Optional[str] = None,
    ):
        """Record a completed simulation."""
        record = SimulationRecord(
            timestamp=datetime.now(),
            user_query=user_query,
            pde_params=pde_params,
            solver_result=solver_result,
            html_path=html_path,
            data_file=data_file,
            summary=summary,
        )
        self.simulation_history.append(record)
        self.current_simulation = record
        
        # Keep only last 10 simulations
        if len(self.simulation_history) > 10:
            self.simulation_history = self.simulation_history[-10:]
    
    def get_context_summary(self) -> str:
        """Get a summary of conversation context for the agent."""
        if not self.simulation_history and not self.messages:
            return ""
        
        context_parts = []
        
        # Add recent simulation summary
        if self.current_simulation:
            context_parts.append("## Most Recent Simulation")
            context_parts.append(f"Query: {self.current_simulation.user_query}")
            if self.current_simulation.pde_params:
                params = self.current_simulation.pde_params
                context_parts.append(f"Type: {params.get('pde_type', 'N/A')} {params.get('dim', 'N/A')}D")
                if params.get('domain_size'):
                    context_parts.append(f"Domain: {params['domain_size']}")
                if params.get('bc_values'):
                    context_parts.append(f"Boundary Conditions: {params['bc_values']}")
                if params.get('initial_value') is not None:
                    context_parts.append(f"Initial Value: {params['initial_value']}")
            if self.current_simulation.html_path:
                context_parts.append(f"Visualization: {self.current_simulation.html_path}")
            context_parts.append("")
        
        # Add previous simulations count
        if len(self.simulation_history) > 1:
            context_parts.append(f"Note: There are {len(self.simulation_history)} previous simulations in this session.")
            context_parts.append("You can reference previous simulations when answering follow-up questions.")
            context_parts.append("")
        
        # Add recent conversation summary
        if len(self.messages) > 2:
            recent_user_msgs = [msg.content for msg in self.messages[-10:] if isinstance(msg, HumanMessage)]
            if recent_user_msgs:
                context_parts.append("## Recent Conversation")
                for i, msg in enumerate(recent_user_msgs[-3:], 1):  # Last 3 user messages
                    context_parts.append(f"{i}. User: {msg[:100]}...")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_messages_for_agent(self, system_prompt: str, include_context: bool = True) -> List[BaseMessage]:
        """Get messages formatted for agent, including context if requested."""
        messages = []
        
        # Add system prompt with context
        if include_context:
            context = self.get_context_summary()
            if context:
                enhanced_prompt = f"{system_prompt}\n\n## Conversation Context\n{context}\n\nUse this context to understand follow-up questions and provide relevant information."
            else:
                enhanced_prompt = system_prompt
        else:
            enhanced_prompt = system_prompt
        
        messages.append(SystemMessage(content=enhanced_prompt))
        
        # Add conversation history (excluding system messages already added)
        for msg in self.messages:
            if not isinstance(msg, SystemMessage):
                messages.append(msg)
        
        return messages
    
    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        self.simulation_history = []
        self.current_simulation = None
    
    def save(self):
        """Save conversation memory to file."""
        if not self.persist_file:
            return
        
        # Check if path exists and is a directory (e.g., Docker volume mount issue)
        if os.path.exists(self.persist_file) and os.path.isdir(self.persist_file):
            print(f"Warning: {self.persist_file} is a directory, cannot save. Skipping persistence.")
            return
        
        data = {
            "messages": [self._message_to_dict(msg) for msg in self.messages],
            "simulation_history": [sim.to_dict() for sim in self.simulation_history],
            "current_simulation": self.current_simulation.to_dict() if self.current_simulation else None,
        }
        
        try:
            with open(self.persist_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except (IOError, OSError) as e:
            print(f"Warning: Failed to save conversation memory to {self.persist_file}: {e}")
    
    def load(self):
        """Load conversation memory from file."""
        if not self.persist_file or not os.path.exists(self.persist_file):
            return
        
        # Check if path is a directory (e.g., Docker volume mount issue)
        if os.path.isdir(self.persist_file):
            print(f"Warning: {self.persist_file} is a directory, cannot load. Skipping persistence.")
            return
        
        try:
            with open(self.persist_file, "r") as f:
                data = json.load(f)
            
            # Load messages (simplified - just recreate human/ai messages)
            self.messages = []
            for msg_dict in data.get("messages", []):
                msg_type = msg_dict.get("type")
                content = msg_dict.get("content", "")
                if msg_type == "human":
                    self.messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    self.messages.append(AIMessage(content=content))
                elif msg_type == "system":
                    self.messages.append(SystemMessage(content=content))
            
            # Load simulation history
            self.simulation_history = []
            for sim_dict in data.get("simulation_history", []):
                record = SimulationRecord(
                    timestamp=datetime.fromisoformat(sim_dict["timestamp"]),
                    user_query=sim_dict["user_query"],
                    pde_params=sim_dict.get("pde_params"),
                    solver_result=sim_dict.get("solver_result"),
                    html_path=sim_dict.get("html_path"),
                    data_file=sim_dict.get("data_file"),
                    summary=sim_dict.get("summary"),
                )
                self.simulation_history.append(record)
            
            # Set current simulation
            if data.get("current_simulation"):
                cs = data["current_simulation"]
                self.current_simulation = SimulationRecord(
                    timestamp=datetime.fromisoformat(cs["timestamp"]),
                    user_query=cs["user_query"],
                    pde_params=cs.get("pde_params"),
                    solver_result=cs.get("solver_result"),
                    html_path=cs.get("html_path"),
                    data_file=cs.get("data_file"),
                    summary=cs.get("summary"),
                )
        except Exception as e:
            print(f"Warning: Failed to load conversation memory: {e}")
    
    @staticmethod
    def _message_to_dict(msg: BaseMessage) -> Dict[str, Any]:
        """Convert a message to dictionary."""
        return {
            "type": msg.__class__.__name__.lower().replace("message", ""),
            "content": msg.content,
        }

