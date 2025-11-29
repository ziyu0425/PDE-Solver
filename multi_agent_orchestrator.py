# multi_agent_orchestrator.py
"""
Multi-Agent Orchestrator
-------------------------
Coordinates PDE Parser Agent and Dispatcher Agent to solve PDE problems from natural language.
Supports conversation memory for context-aware follow-up questions.
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pde_parser_agent import PDEParserAgent
from dispatcher_agent import DispatcherAgent
from pde_schema import PDEParameters
from conversation_memory import ConversationMemory

# Load environment variables from .env file
load_dotenv()

# Absolute path to the FEniCS MCP server
FENICS_SERVER_PATH = os.path.abspath("fenics_mcp_server.py")


class MultiAgentOrchestrator:
    """Orchestrates PDE Parser and Dispatcher agents with conversation memory."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        memory_file: Optional[str] = "conversation_memory.json"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model_name: Model name for LLM agents.
            temperature: Temperature for LLM agents.
            memory_file: Optional file path to persist conversation memory.
        """
        # Ensure environment variables are loaded
        load_dotenv()
        
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not found. Please set it in your environment "
                "or create a .env file with OPENAI_API_KEY=your_key"
            )
        
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.mcp_client = None
        self.parser_agent = None
        self.dispatcher_agent = None
        self.memory = ConversationMemory(persist_file=memory_file)
        self._initialized = False
    
    async def initialize(self):
        """Initialize MCP client and agents."""
        if self._initialized:
            return
        
        # Initialize MCP client
        self.mcp_client = MultiServerMCPClient(
            {
                "fenics_fem": {
                    "command": "python",
                    "args": [FENICS_SERVER_PATH],
                    "transport": "stdio",
                }
            }
        )
        
        # Initialize agents
        self.parser_agent = PDEParserAgent(llm=self.llm)
        self.dispatcher_agent = DispatcherAgent(mcp_client=self.mcp_client, llm=self.llm)
        
        self._initialized = True
    
    async def solve(self, description: str) -> Dict[str, Any]:
        """
        Solve a PDE problem from natural language description.
        Uses conversation memory to handle follow-up questions.
        
        Args:
            description: Natural language description of the PDE problem.
            
        Returns:
            Dictionary with:
            - "pde_params": Parsed PDEParameters
            - "dispatch_result": Result from dispatcher
            - "summary": Overall summary
        """
        if not self._initialized:
            await self.initialize()
        
        # Check if this is a greeting or non-PDE query
        if self._is_greeting_or_non_pde_query(description):
            greeting_response = self._handle_greeting(description)
            self.memory.add_user_message(description)
            self.memory.add_ai_message(greeting_response)
            self.memory.save()
            return {
                "response": greeting_response,
                "html_path": None,
                "data_file": None,
                "status": "greeting",
                "summary": greeting_response  # Also include in summary for consistency
            }
        
        # Add user message to memory
                # Add user message to memory
        self.memory.add_user_message(description)

        # 1. 先判断是不是 follow-up，并取出上一轮仿真
        is_followup = self._is_followup_question(description)
        prev_params = None
        prev_query = None
        if self.memory.current_simulation:
            if self.memory.current_simulation.pde_params:
                prev_params = self.memory.current_simulation.pde_params
            if self.memory.current_simulation.user_query:
                prev_query = self.memory.current_simulation.user_query

        # 2. 决定要不要做 LLM 验证
        if is_followup and prev_params:
            # ✅ 有上一轮仿真 + 被判定为 follow-up：直接当 PDE 处理，完全跳过 _validate_pde_query
            is_pde_query = True
            justification = "Follow-up query modifying an existing PDE simulation."
            print("   Follow-up query detected - skipping PDE validation and treating as PDE.")
            if prev_query:
                print(f"      Previous query: {prev_query}")
            print(f"      Current query: {description}")
        else:
            # 只有“非 follow-up 的新问题”才做严格验证（只调用一次）
            is_pde_query, justification = await self._validate_pde_query(description)
            print(f"   Validation result: is_pde={is_pde_query}, justification={justification}")

        if not is_pde_query:
            # Not a PDE problem - provide helpful response
            non_pde_response = (
                "I understand your query, but this doesn't appear to be a PDE (Partial Differential Equation) problem that I can solve.\n\n"
                f"**Reasoning:** {justification}\n\n"
                "**What I can help with:**\n"
                "- Heat transfer problems (1D, 2D, 3D)\n"
                "- Elasticity problems (1D, 2D, 3D)\n"
                "- Transient or steady-state simulations\n\n"
                "**Example queries:**\n"
                "- \"Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C\"\n"
                "- \"Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa\"\n"
                "- \"Simulate heat diffusion in a 1m x 1m plate with initial temperature 10°C\"\n\n"
                "Please describe a PDE problem you'd like me to solve!"
            )
            self.memory.add_ai_message(non_pde_response)
            self.memory.save()
            return {
                "response": non_pde_response,
                "html_path": None,
                "data_file": None,
                "status": "not_pde",
                "summary": non_pde_response,
                "justification": justification
            }
        
        # Step 1: Detect if this is a follow-up question
        #is_followup = self._is_followup_question(description)
        #prev_params = None
        #if self.memory.current_simulation and self.memory.current_simulation.pde_params:
        #    prev_params = self.memory.current_simulation.pde_params
        
        # Step 2: Parse natural language to structured parameters
        # For follow-ups: tell parser to ONLY extract changes, preserve everything else
        # For new queries: parse everything from scratch
        enhanced_description = description
        if is_followup and prev_params:
            # For follow-ups, provide full previous parameters and explicit instructions
            prev_params_str = self._format_previous_params(prev_params)
            enhanced_description = (
                f"FOLLOW-UP MODIFICATION REQUEST:\n"
                f"User request: {description}\n\n"
                f"CURRENT SIMULATION PARAMETERS (preserve ALL of these unless explicitly changed):\n"
                f"{prev_params_str}\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- This is a MODIFICATION of the previous simulation\n"
                f"- Extract ONLY the parameters that are EXPLICITLY mentioned in the user's request\n"
                f"- Return ALL other parameters as null/None/empty so they can be preserved\n"
                f"- DO NOT change pde_type, dim, domain_size, boundary conditions, etc. unless mentioned\n"
                f"- Example: If user says 'add heat source 10', return ONLY: {{\"source_type\": \"constant\", \"source_value\": 10.0}}\n"
                f"- Leave everything else null/empty"
            )
        
        print("🔍 Parsing PDE problem description...")
        if is_followup:
            print("   (Detected as follow-up - will preserve previous parameters)")
        try:
            pde_params = await self.parser_agent.parse(enhanced_description)
            #try:
            #    print("[DEBUG] Parsed PDEParameters:")
            #    print(f"   pde_type      = {pde_params.pde_type}")
            #    print(f"   dim           = {pde_params.dim}")
            #    print(f"   geometry_type = {pde_params.geometry_type}")
            #    print(f"   coordinate_system = {getattr(pde_params, 'coordinate_system', None)}")
            #    print(f"   domain_size   = {pde_params.domain_size}")
            #    print(f"   bc_values     = {pde_params.bc_values}")
            #    print(f"   geometry_params = {getattr(pde_params, 'geometry_params', None)}")
            #    print(f"   diffusivity   = {pde_params.diffusivity}")
            #except Exception as debug_e:
            #    print(f"[DEBUG] Failed to print PDEParameters: {debug_e}")
            # Normalize domain_size immediately after parsing (fix malformed formats like {"domain_size": value})
            # Pass the original description for fallback extraction if parser missed dimensions
            if pde_params.domain_size:
                pde_params.domain_size = self._normalize_domain_size(
                    pde_params.domain_size, 
                    pde_params.dim,
                    description
                )
            
            # Step 3: If this is a follow-up, merge with previous parameters
            # CRITICAL: Start with ALL previous parameters, only override what was explicitly mentioned
            if is_followup and prev_params:
                print("   Merging with previous parameters...")
                pde_params = self._merge_parameters(prev_params, pde_params, description)
            elif not is_followup:
                # New query - ensure we have required parameters with defaults
                if not pde_params.domain_size:
                    # Set reasonable defaults for new queries
                    if pde_params.dim == 1:
                        pde_params.domain_size = {"length": 2.0}
                    elif pde_params.dim == 2:
                        pde_params.domain_size = {"Lx": 1.0, "Ly": 1.0}
                    elif pde_params.dim == 3:
                        pde_params.domain_size = {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0}
            
            print(f"✅ Parsed: {pde_params.pde_type} {pde_params.dim}D problem")
            print(f"   Domain: {pde_params.domain_size}")
            print(f"   Boundary conditions: {pde_params.bc_values}")
            if pde_params.notes:
                print(f"   Notes: {', '.join(pde_params.notes)}")
        except Exception as e:
            # Add error message to memory
            error_msg = f"Failed to parse PDE description: {e}"
            self.memory.add_ai_message(f"Error: {error_msg}")
            return {
                "error": error_msg,
                "description": description,
            }
        
        # Step 2: Dispatch to MCP tools
        print(f"\n🚀 Dispatching to {pde_params.pde_type} {pde_params.dim}D solver...")
        if is_followup:
            print(f"   (Using merged parameters - summary will reflect all current values)")
        try:
            dispatch_result = await self.dispatcher_agent.dispatch(pde_params)
            
            if "error" in dispatch_result:
                error_msg = dispatch_result["error"]
                self.memory.add_ai_message(f"Error: {error_msg}")
                return {
                    "error": error_msg,
                    "pde_params": pde_params,
                    "solver_args": dispatch_result.get("solver_args"),
                }
            
            # Record successful simulation in memory
            # Store COMPLETE parameters (after merge) for future follow-ups
            pde_params_dict = {
                "pde_type": pde_params.pde_type,
                "dim": pde_params.dim,
                "domain_size": pde_params.domain_size,
                "nx": pde_params.nx,
                "ny": pde_params.ny,
                "nz": pde_params.nz,
                "bc_values": pde_params.bc_values,
                "initial_value": pde_params.initial_value,
                "initial_type": pde_params.initial_type or "constant",
                "initial_amplitude": pde_params.initial_amplitude if pde_params.initial_amplitude is not None else 1.0,
                "initial_wavenumber": pde_params.initial_wavenumber if pde_params.initial_wavenumber is not None else 1.0,
                "diffusivity": pde_params.diffusivity,
                "young_modulus": pde_params.young_modulus,
                "poisson_ratio": pde_params.poisson_ratio,
                "density": pde_params.density,
                "material_params": pde_params.material_params,
                "dt": pde_params.dt,
                "num_steps": pde_params.num_steps,
                "total_time": pde_params.total_time,
                "source_type": pde_params.source_type or "none",
                "source_value": pde_params.source_value if pde_params.source_value is not None else 0.0,
                "steady": pde_params.steady if pde_params.steady is not None else False,
                "field_name": pde_params.field_name,
                "unit": pde_params.unit,
            }
            
            self.memory.record_simulation(
                user_query=description,
                pde_params=pde_params_dict,
                solver_result=dispatch_result.get("solver_result"),
                html_path=dispatch_result.get("html_path"),
                data_file=dispatch_result.get("data_file"),
                summary=dispatch_result.get("summary"),
            )
            
            # Add success message to memory
            summary = dispatch_result.get("summary", "Simulation completed successfully.")
            self.memory.add_ai_message(summary)
            
            # Save memory to file
            self.memory.save()
            
            return {
                "pde_params": pde_params,
                "dispatch_result": dispatch_result,
                "summary": summary,
                "html_path": dispatch_result.get("html_path"),
                "data_file": dispatch_result.get("data_file"),
            }
        except Exception as e:
            error_msg = f"Failed to dispatch and solve: {e}"
            self.memory.add_ai_message(f"Error: {error_msg}")
            return {
                "error": error_msg,
                "pde_params": pde_params,
            }
    
    def _is_greeting_or_non_pde_query(self, description: str) -> bool:
        """
        Detect if the description is a greeting or casual conversation, not a PDE problem.
        Returns True if it's a greeting/non-PDE query, False if it's a PDE problem.
        CRITICAL: PDE keywords take priority - if PDE keywords are present, treat as PDE query.
        """
        description_lower = description.lower().strip()
        
        # PDE keywords - if ANY of these are present, treat as PDE query (not greeting)
        pde_keywords = [
            "heat", "temperature", "diffusion", "conduction", "transfer",
            "elasticity", "stress", "strain", "displacement", "force",
            "solve", "simulate", "simulation", "pde", "equation",
            "rod", "bar", "plate", "cube", "domain", "boundary",
            "1d", "2d", "3d", "one-dimensional", "two-dimensional", "three-dimensional",
            "initial", "boundary", "condition", "young", "modulus", "poisson", "ratio",
            "length", "meter", "m ", "lx", "ly", "lz", "width", "height", "depth",
            "celsius", "c ", "kelvin", "pa", "gpa", "source", "steady", "transient"
        ]
        
        # CRITICAL: Check for PDE keywords FIRST - if found, it's NOT a greeting
        has_pde_keyword = any(keyword in description_lower for keyword in pde_keywords)
        if has_pde_keyword:
            return False  # It's a PDE query, not a greeting
        
        # Greetings and casual conversation (only check if no PDE keywords found)
        greetings = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
            "what's up", "whats up", "how are you", "how do you do", "nice to meet you",
            "thanks", "thank you", "bye", "goodbye", "see you", "see ya"
        ]
        
        # Very short messages that are likely greetings (only if no PDE keywords)
        if len(description_lower) <= 5 and any(greeting in description_lower for greeting in ["hi", "hey", "bye", "ok", "yes", "no", "okay"]):
            return True
        
        # Check for greetings (only if no PDE keywords)
        if any(greeting in description_lower for greeting in greetings):
            return True
        
        return False
    
    async def _validate_pde_query(self, description: str) -> Tuple[bool, str]:
        """
        Use LLM to validate whether the query is actually a PDE problem.
        Returns (is_pde_query: bool, justification: str).
        """
        if not self._initialized:
            await self.initialize()
        
        validation_prompt = f"""You are a PDE (Partial Differential Equation) problem validator.

Your task is to determine whether the following user query describes a PDE problem that can be solved using numerical methods.

PDE problems typically involve:
- Heat transfer/diffusion (temperature fields)
- Elasticity (stress, strain, displacement)
- Wave equations
- Fluid flow (with appropriate simplification)
- Other field equations solved over spatial domains with boundary conditions

Examples of VALID PDE problems:
- "Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C"
- "Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa"
- "Simulate heat diffusion in a 1m x 1m plate with initial temperature 10°C"
- "3D elasticity problem on a 1m x 0.2m x 0.2m cube with gravity"

Examples of INVALID (NOT PDE problems):
- "What is the weather today?"
- "Calculate 2 + 2"
- "Tell me a joke"
- "How do I cook pasta?"
- General questions that don't involve solving PDEs

User query: "{description}"

Respond with a JSON object:
{{
    "is_pde_problem": true/false,
    "justification": "Brief explanation of why this is or isn't a PDE problem (1-2 sentences)"
}}

Be strict: only return true if the query clearly describes a PDE problem to be solved. If it's ambiguous, unclear, or not a PDE problem, return false."""

        try:
            messages = [
                SystemMessage(content="You are a PDE problem validator. Always respond with valid JSON."),
                HumanMessage(content=validation_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Try to parse JSON from response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            validation_result = json.loads(response_text)
            
            is_pde = validation_result.get("is_pde_problem", False)
            justification = validation_result.get("justification", "No justification provided.")
            
            return (is_pde, justification)
        except Exception as e:
            # If validation fails, default to treating as PDE problem (lenient approach)
            print(f"Warning: PDE validation failed: {e}. Proceeding with assumption that it's a PDE problem.")
            return (True, "Validation error - proceeding with PDE assumption.")
    
    def _handle_greeting(self, description: str) -> str:
        """Generate an appropriate response for greetings and non-PDE queries."""
        description_lower = description.lower().strip()
        
        if any(word in description_lower for word in ["hi", "hello", "hey"]):
            return (
                "Hello! 👋 I'm a PDE Solver chatbot. I can help you solve partial differential equations (PDEs) "
                "using natural language.\n\n"
                "**What I can do:**\n"
                "- Solve heat transfer problems (1D, 2D, 3D)\n"
                "- Solve elasticity problems (1D, 2D, 3D)\n"
                "- Handle transient and steady-state simulations\n\n"
                "**Try asking me:**\n"
                "- \"Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C\"\n"
                "- \"Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa\"\n"
                "- \"Simulate heat diffusion in a 1m x 1m plate with initial temperature 10°C\"\n\n"
                "Just describe your PDE problem in natural language!"
            )
        elif any(word in description_lower for word in ["thanks", "thank you"]):
            return "You're welcome! Feel free to ask me any PDE problems you'd like to solve. 😊"
        elif any(word in description_lower for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! Feel free to come back anytime with your PDE problems. 👋"
        else:
            return (
                "Hi! I'm a PDE Solver chatbot. I can help you solve partial differential equations (PDEs).\n\n"
                "**Try asking me to solve a PDE problem, for example:**\n"
                "- \"Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C\"\n"
                "- \"Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa\"\n\n"
                "Just describe your problem in natural language!"
            )
    
    def _is_followup_question(self, description: str) -> bool:
        """
        Detect if the description is a follow-up question.
        Returns True if there's a previous simulation AND the query seems to modify it.
        """
        # If no previous simulation, this can't be a follow-up
        if not self.memory.current_simulation:
            return False
        
        description_lower = description.lower().strip()
        
        # Explicit follow-up keywords (more comprehensive)
        followup_keywords = [
            "change", "modify", "update", "different", "same", "again", 
            "repeat", "rerun", "previous", "last", "before", "instead",
            "with", "without", "adjust", "set", "make", "add", "remove",
            "increase", "decrease", "new", "also", "too", "and"
        ]
        
        # Check if description contains follow-up keywords
        if any(keyword in description_lower for keyword in followup_keywords):
            return True
        
        # Check if description is very short (likely a modification)
        # Examples: "heat source 10", "boundary 50", "dt 0.001"
        if len(description.split()) <= 6 and self.memory.current_simulation:
            return True
        
        # If it doesn't contain domain/geometry information but we have a previous sim, likely follow-up
        geometry_keywords = ["rod", "bar", "plate", "cube", "domain", "length", "meter", "m ", "lx", "ly", "lz", "width", "height", "depth"]
        has_geometry = any(keyword in description_lower for keyword in geometry_keywords)
        if not has_geometry and self.memory.current_simulation:
            # No geometry mentioned but we have previous sim - likely a modification
            return True
        
        return False
    
    def _format_previous_params(self, prev_params: Dict[str, Any]) -> str:
        """Format previous parameters as a readable string for the parser."""
        lines = []
        lines.append("Previous simulation parameters (preserve these unless explicitly changed):")
        lines.append("")
        lines.append(f"pde_type: {prev_params.get('pde_type', 'heat')}")
        lines.append(f"dim: {prev_params.get('dim', 1)}")
        if prev_params.get('domain_size'):
            lines.append(f"domain_size: {prev_params['domain_size']}")
        if prev_params.get('nx') is not None:
            lines.append(f"nx: {prev_params['nx']}")
        if prev_params.get('ny') is not None:
            lines.append(f"ny: {prev_params['ny']}")
        if prev_params.get('nz') is not None:
            lines.append(f"nz: {prev_params['nz']}")
        if prev_params.get('diffusivity') is not None:
            lines.append(f"diffusivity: {prev_params['diffusivity']}")
        if prev_params.get('young_modulus') is not None:
            lines.append(f"young_modulus: {prev_params['young_modulus']}")
        if prev_params.get('poisson_ratio') is not None:
            lines.append(f"poisson_ratio: {prev_params['poisson_ratio']}")
        if prev_params.get('density') is not None:
            lines.append(f"density: {prev_params['density']}")
        if prev_params.get('material_params'):
            lines.append(f"material_params: {prev_params['material_params']}")
        if prev_params.get('bc_values'):
            lines.append(f"bc_values: {prev_params['bc_values']}")
        if prev_params.get('initial_value') is not None:
            lines.append(f"initial_value: {prev_params['initial_value']}")
        if prev_params.get('initial_type'):
            lines.append(f"initial_type: {prev_params['initial_type']}")
        if prev_params.get('initial_amplitude') is not None:
            lines.append(f"initial_amplitude: {prev_params['initial_amplitude']}")
        if prev_params.get('initial_wavenumber') is not None:
            lines.append(f"initial_wavenumber: {prev_params['initial_wavenumber']}")
        if prev_params.get('source_type'):
            lines.append(f"source_type: {prev_params['source_type']}")
        if prev_params.get('source_value') is not None:
            lines.append(f"source_value: {prev_params['source_value']}")
        if prev_params.get('steady') is not None:
            lines.append(f"steady: {prev_params['steady']}")
        if prev_params.get('dt') is not None:
            lines.append(f"dt: {prev_params['dt']}")
        if prev_params.get('num_steps') is not None:
            lines.append(f"num_steps: {prev_params['num_steps']}")
        if prev_params.get('total_time') is not None:
            lines.append(f"total_time: {prev_params['total_time']}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("CRITICAL INSTRUCTIONS FOR PARSER:")
        lines.append("=" * 60)
        lines.append("1. This is a FOLLOW-UP modification request")
        lines.append("2. Extract ONLY parameters explicitly mentioned in the user's request")
        lines.append("3. For ALL other parameters, return null/None/empty - DO NOT include them")
        lines.append("4. DO NOT return pde_type, dim, or domain_size unless the user explicitly changes them")
        lines.append("5. Examples:")
        lines.append("   - 'add heat source 10' → Return ONLY: {\"source_type\": \"constant\", \"source_value\": 10.0}")
        lines.append("   - 'change boundary to 50' → Return ONLY: {\"bc_values\": {\"t_left\": 50}}")
        lines.append("   - 'different initial temperature 100' → Return ONLY: {\"initial_value\": 100}")
        lines.append("6. If you return domain_size, dim, or pde_type when not mentioned, it will OVERWRITE the previous simulation")
        lines.append("7. The system will merge your changes with the previous parameters above")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _normalize_domain_size(self, domain_size: Dict[str, Any], dim: int, description: str = "") -> Dict[str, float]:
        """
        Normalize domain_size from malformed formats (e.g., {"domain_size": value}) 
        to the correct format based on dimension.
        If the parser only extracted a single value, try to extract all dimensions from the description.
        Also handles cases where "thick" or "thickness" was incorrectly used as length.
        
        Args:
            domain_size: The domain_size dictionary (may be malformed)
            dim: The dimension (1, 2, or 3)
            description: Original query description (optional, for fallback extraction)
            
        Returns:
            Normalized domain_size dictionary
        """
        if not domain_size:
            return {}
        
        # Check for "thick" or "thickness" in description - this should NOT be used as length
        description_lower = description.lower() if description else ""
        has_thick = "thick" in description_lower or "thickness" in description_lower
        
        # Check if it's malformed (has nested "domain_size" key)
        if "domain_size" in domain_size and isinstance(domain_size["domain_size"], (int, float)):
            value = float(domain_size["domain_size"])
            
            # For 1D: if value is extremely small (< 1e-6) and description mentions "thick", use default length
            if dim == 1 and has_thick and value < 1e-6:
                # This is likely thickness, not length - use default length
                return {"length": 2.0}
            
            # Try to extract all dimensions from the description if it's 2D or 3D
            if dim == 2 and description:
                # Look for patterns like "1m x 1m", "1m*1m", etc.
                import re
                # Match patterns like "number unit x number unit" or "number unit * number unit"
                pattern = r'(\d+(?:\.\d+)?)\s*m\s*[x\*×]\s*(\d+(?:\.\d+)?)\s*m'
                matches = re.findall(pattern, description, re.IGNORECASE)
                if matches:
                    first_dim, second_dim = matches[0]
                    return {"Lx": float(first_dim), "Ly": float(second_dim)}
            
            elif dim == 3 and description:
                # Look for patterns like "1m x 0.2m x 0.2m", "1m*0.2m*0.2m", "1m x 0.2m * 0.2m"
                import re
                # Match patterns with three dimensions
                pattern = r'(\d+(?:\.\d+)?)\s*m\s*[x\*×]\s*(\d+(?:\.\d+)?)\s*m\s*[x\*×]\s*(\d+(?:\.\d+)?)\s*m'
                matches = re.findall(pattern, description, re.IGNORECASE)
                if matches:
                    first_dim, second_dim, third_dim = matches[0]
                    return {"Lx": float(first_dim), "Ly": float(second_dim), "Lz": float(third_dim)}
                
                # Try pattern with asterisk and spaces: "1m x 0.2m * 0.2m"
                pattern2 = r'(\d+(?:\.\d+)?)\s*m\s*[x\*×]\s*(\d+(?:\.\d+)?)\s*m\s*\*\s*(\d+(?:\.\d+)?)\s*m'
                matches2 = re.findall(pattern2, description, re.IGNORECASE)
                if matches2:
                    first_dim, second_dim, third_dim = matches2[0]
                    return {"Lx": float(first_dim), "Ly": float(second_dim), "Lz": float(third_dim)}
            
            # Fallback: use the single value for all dimensions
            if dim == 1:
                return {"length": value}
            elif dim == 2:
                return {"Lx": value, "Ly": value}
            elif dim == 3:
                return {"Lx": value, "Ly": value, "Lz": value}
            else:
                return {}
        
        # Already in correct format, return as-is
        return domain_size
    
    def _merge_parameters(
        self, 
        prev_params: Dict[str, Any], 
        new_params: PDEParameters,
        description: str
    ) -> PDEParameters:
        """
        Merge previous parameters with new ones.
        Only override parameters that are explicitly mentioned in the description.
        """
        description_lower = description.lower()
        
        # Start with previous parameters
        merged = PDEParameters()
        
        # Copy all previous parameters
        merged.pde_type = prev_params.get('pde_type', 'heat')
        merged.dim = prev_params.get('dim', 1)
        merged.domain_size = prev_params.get('domain_size', {}).copy() if prev_params.get('domain_size') else {}
        merged.nx = prev_params.get('nx')
        merged.ny = prev_params.get('ny')
        merged.nz = prev_params.get('nz')
        merged.diffusivity = prev_params.get('diffusivity')
        merged.young_modulus = prev_params.get('young_modulus')
        merged.poisson_ratio = prev_params.get('poisson_ratio')
        merged.density = prev_params.get('density')
        merged.material_params = prev_params.get('material_params', {}).copy() if prev_params.get('material_params') else {}
        merged.bc_values = prev_params.get('bc_values', {}).copy() if prev_params.get('bc_values') else {}
        merged.initial_value = prev_params.get('initial_value')
        merged.initial_type = prev_params.get('initial_type', 'constant')
        merged.initial_amplitude = prev_params.get('initial_amplitude', 1.0)
        merged.initial_wavenumber = prev_params.get('initial_wavenumber', 1.0)
        merged.source_type = prev_params.get('source_type', 'none')
        merged.source_value = prev_params.get('source_value', 0.0)
        merged.steady = prev_params.get('steady', False)
        merged.dt = prev_params.get('dt')
        merged.num_steps = prev_params.get('num_steps')
        merged.field_name = prev_params.get('field_name', 'temperature')
        merged.unit = prev_params.get('unit', '°C')
        
        # CRITICAL: Override ONLY if new_params has a non-None/non-empty value
        # This ensures we only change what was explicitly mentioned in the follow-up query
        # ALL other parameters remain from previous simulation
        
        # PDE type and dimension - preserve from previous unless explicitly changed
        # Check description for explicit mentions to avoid false overrides from parser defaults
        
        description_lower = description.lower()
        prev_dim = prev_params.get('dim', 1)
        
        # Check if dimension was explicitly mentioned in description
        dim_keywords_1d = ["1d", "1-d", "one-dimensional", "one dimensional", "line", "rod", "bar", "1 dimension"]
        dim_keywords_2d = ["2d", "2-d", "two-dimensional", "two dimensional", "plate", "sheet", "2 dimension"]
        dim_keywords_3d = ["3d", "3-d", "three-dimensional", "three dimensional", "cube", "box", "3 dimension"]
        
        has_1d_mention = any(keyword in description_lower for keyword in dim_keywords_1d)
        has_2d_mention = any(keyword in description_lower for keyword in dim_keywords_2d)
        has_3d_mention = any(keyword in description_lower for keyword in dim_keywords_3d)
        
        # Dimension handling: only override if explicitly mentioned in description
        if has_1d_mention:
            # User explicitly mentioned 1D - override to 1D
            merged.dim = 1
        elif has_2d_mention:
            # User explicitly mentioned 2D - override to 2D
            merged.dim = 2
        elif has_3d_mention:
            # User explicitly mentioned 3D - override to 3D
            merged.dim = 3
        else:
            # No dimension keywords in description - preserve previous dimension
            # Be very conservative: only override if new_params.dim is different AND previous was 1D
            # (If previous was 2D/3D and no dimension keywords, definitely preserve it)
            if prev_dim == 1:
                # Previous was 1D - allow override to 2D/3D if parser extracted it
                if new_params.dim is not None and new_params.dim != 1:
                    merged.dim = new_params.dim
                # Otherwise keep dim=1
            else:
                # Previous was 2D or 3D - preserve it unless dimension was explicitly mentioned
                # (We already checked for keywords above, so preserve previous)
                pass  # Keep merged.dim from previous (already set above)
        
        # Check if PDE type was explicitly mentioned
        pde_mentions = ["wave equation", "wave", "advection", "poisson", "laplace", "elasticity"]
        has_pde_mention = any(mention in description_lower for mention in pde_mentions)
        if has_pde_mention and new_params.pde_type and new_params.pde_type != "heat":
            merged.pde_type = new_params.pde_type
        # Don't let default value override previous
        
        # Domain parameters - only override if provided
        # Normalize domain_size format if it's malformed (e.g., {"domain_size": value})
        if new_params.domain_size and len(new_params.domain_size) > 0:
            # Use new_params.dim if it's explicitly set, otherwise use merged.dim
            dim_for_normalization = new_params.dim if new_params.dim is not None else merged.dim
            normalized_domain = self._normalize_domain_size(new_params.domain_size, dim_for_normalization, description)
            merged.domain_size = normalized_domain
        if new_params.nx is not None:
            merged.nx = new_params.nx
        if new_params.ny is not None:
            merged.ny = new_params.ny
        if new_params.nz is not None:
            merged.nz = new_params.nz
        
        # Boundary conditions - merge dict if provided (update only mentioned boundaries)
        if new_params.bc_values and len(new_params.bc_values) > 0:
            merged.bc_values.update(new_params.bc_values)
        
        # Initial condition - only override if explicitly set
        if new_params.initial_value is not None:
            merged.initial_value = new_params.initial_value
        if new_params.initial_type:
            merged.initial_type = new_params.initial_type
        if new_params.initial_amplitude is not None:
            merged.initial_amplitude = new_params.initial_amplitude
        if new_params.initial_wavenumber is not None:
            merged.initial_wavenumber = new_params.initial_wavenumber
        
        # Source term - only override if mentioned
        if new_params.source_type and new_params.source_type != "none":
            merged.source_type = new_params.source_type
        if new_params.source_value is not None:
            merged.source_value = new_params.source_value
        
        # Steady-state mode - only override if explicitly set
        if new_params.steady is not None:
            merged.steady = new_params.steady
        
        # Time discretization - only override if mentioned
        if new_params.dt is not None:
            merged.dt = new_params.dt
        if new_params.num_steps is not None:
            merged.num_steps = new_params.num_steps
        if new_params.total_time is not None:
            merged.total_time = new_params.total_time
        
        # Material parameters - only override if mentioned
        if new_params.diffusivity is not None:
            merged.diffusivity = new_params.diffusivity
        if new_params.young_modulus is not None:
            merged.young_modulus = new_params.young_modulus
        if new_params.poisson_ratio is not None:
            merged.poisson_ratio = new_params.poisson_ratio
        if new_params.density is not None:
            merged.density = new_params.density
        if new_params.material_params and len(new_params.material_params) > 0:
            # Merge material_params dict (update only mentioned parameters)
            merged.material_params.update(new_params.material_params)
        
        # Update metadata
        if new_params.field_name:
            merged.field_name = new_params.field_name
        if new_params.unit:
            merged.unit = new_params.unit
        
        # Add note about what was changed
        changed_params = []
        if new_params.domain_size:
            changed_params.append("domain")
        if new_params.bc_values:
            changed_params.append("boundary conditions")
        if new_params.initial_value is not None:
            changed_params.append("initial condition")
        if new_params.source_type or new_params.source_value is not None:
            changed_params.append("source term")
        if new_params.steady is not None:
            changed_params.append("steady-state mode")
        
        if changed_params:
            merged.notes = [f"Modified: {', '.join(changed_params)}. All other parameters preserved from previous simulation."]
        else:
            merged.notes = ["All parameters preserved from previous simulation."]
        
        return merged
    
    def get_memory(self) -> ConversationMemory:
        """Get the conversation memory object."""
        return self.memory
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        self.memory.save()


async def run_chat_loop():
    """Run an interactive chat loop with the multi-agent system."""
    # load_dotenv() is already called at module level and in __init__
    orchestrator = MultiAgentOrchestrator()
    
    print("=" * 60)
    print("Multi-Agent PDE Solver (with Memory)")
    print("=" * 60)
    print("\nThis system uses two specialized agents:")
    print("1. PDE Parser Agent: Extracts structured parameters from natural language")
    print("2. Dispatcher Agent: Calls FEniCS MCP tools to solve the problem")
    print("\n✨ Features:")
    print("  - Remembers previous simulations")
    print("  - Supports follow-up questions")
    print("  - Can modify previous parameters")
    print("\nExample queries:")
    print("\nHeat equation examples:")
    print('  "Simulate heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C"')
    print('  "Solve 1D transient heat diffusion with length 1m, initial temperature 100°C"')
    print('  "2D heat equation on a 1m x 1m plate, boundary at 0°C, initial at 20°C"')
    print("\nElasticity examples:")
    print('  "Solve 2D elasticity problem on a 1m x 1m plate with Young\'s modulus 210 GPa"')
    print('  "1D bar elasticity with length 2m, Young\'s modulus 70 GPa (aluminum)"')
    print('  "2D elasticity on a 1m x 1m plate with gravity"')
    print('  "3D elasticity problem on a 1m x 1m x 1m cube, steel material"')
    print('  "2D elasticity with Poisson\'s ratio 0.3, output strain instead of stress"')
    print("\nFollow-up examples:")
    print('  "Change the left boundary temperature to 50°C"')
    print('  "Run the same simulation with different initial temperature"')
    print('  "Add gravity to the elasticity problem"')
    print('  "Change Young\'s modulus to 70 GPa"')
    print('  "Show me the previous results again"')
    print("\nCommands:")
    print("  'exit' or 'quit' - Exit the program")
    print("  'clear' - Clear conversation memory")
    print("  'history' - Show recent simulation history")
    print("\nType 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Clear memory when exiting via Ctrl+C
            try:
                orchestrator.clear_memory()
                print("\n✅ Conversation memory cleared.\n")
            except:
                pass
            break
        
        if user_input.lower() in {"exit", "quit"}:
            # Clear memory before exiting
            orchestrator.clear_memory()
            print("✅ Conversation memory cleared.\n")
            break
        if not user_input:
            continue
        
        # Handle special commands
        if user_input.lower() == "clear":
            orchestrator.clear_memory()
            print("✅ Conversation memory cleared.\n")
            continue
        
        if user_input.lower() == "history":
            memory = orchestrator.get_memory()
            if memory.simulation_history:
                print("\n" + "=" * 60)
                print("SIMULATION HISTORY")
                print("=" * 60)
                for i, sim in enumerate(memory.simulation_history[-5:], 1):  # Last 5
                    print(f"\n{i}. {sim.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Query: {sim.user_query}")
                    if sim.pde_params:
                        print(f"   Type: {sim.pde_params.get('pde_type')} {sim.pde_params.get('dim')}D")
                    if sim.html_path:
                        print(f"   Visualization: {sim.html_path}")
                print("=" * 60 + "\n")
            else:
                print("No simulation history yet.\n")
            continue
        
        print()  # Empty line for readability
        result = await orchestrator.solve(user_input)
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}\n")
            if "pde_params" in result:
                print(f"Parsed parameters: {result['pde_params']}\n")
        else:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(result.get("summary", ""))
            if result.get("html_path"):
                print(f"\n📊 Visualization saved to: {result['html_path']}")
                print("   Open this file in a web browser to view the interactive plot.")
            print("=" * 60 + "\n")
    
    # Clear memory before exiting (in case user exited via Ctrl+C)
    try:
        orchestrator.clear_memory()
    except:
        pass
    
    print("\n👋 Multi-Agent PDE Solver terminated.")


if __name__ == "__main__":
    asyncio.run(run_chat_loop())

