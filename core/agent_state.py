from typing import TypedDict, Any, Optional, List
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    #Input/Output
    current_input: Optional[Any] = None
    final_output: Optional[str] = None
    intermediate_steps: List[tuple] = []
    ##Routing and context
    user_role: Optional[str] = None
    determined_intent: Optional[str] = None
    supervisor_name: Optional[str] = None
    tools_for_agent: List[BaseTool] = []
    #image related
    has_image: bool = False
    image_path: Optional[str] = None
    image_type: Optional[str] = None
    #validation related
    needs_human_validation: bool = False
    validation_result: Optional[str] = None
    validation_comments: Optional[str] = None
    #RAG related
    retrieval_confidence: Optional[float] = None
    rag_sources: List[Any] = None
    web_search_results: Optional[str] = None
    #output from the patient conversational LLM node
    patient_response_text: Optional[str] = None
    #Flag set by the patient LLM node if scheduling is needed  
    request_scheduling: bool = False
    #Field set by the supervisor if it needs to hand off to another supervisor
    next_supervisor_required: Optional[str] = None
    