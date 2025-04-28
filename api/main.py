from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import uuid
from agents.main_orchestrator import build_main_orchestrator_graph  
from core.agent_state import AgentState
from langchain_core.messages import HumanMessage
from config.settings import settings
from core.mcp_manager import MCPToolManager
from core.config_loader import load_mcp_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application startup...")

    mcp_server_configs = load_mcp_config(config_path="config/mcp_servers.json")

    tool_manager = MCPToolManager(mcp_server_configs)
    app.state.tool_manager = tool_manager
    try:
        await app.state.tool_manager.start_client()
        if not app.state.tool_manager.is_running and mcp_server_configs:
            logger.error(
                "MCP Tool Manager failed to start properly during application startup"
            )
        else:
            logger.info("MCP Tool Manager started successfully")
    except Exception as e:
        logger.error(f"Critical error during MCP Manager startup: {e}", exc_info=True)

    try:
        app.state.main_graph = build_main_orchestrator_graph()
        logger.info("stored main graph in app state")
    except Exception as e:
        logger.error(
            f"Critical error building main orchestrator graph: {e}", exc_info=True
        )
        app.state.main_graph = None

    yield

    logger.info("Application shutdown...")
    if hasattr(app.state, "tool_manager") and app.state.tool_manager.is_running:
        logger.info("Stopping MCP Tool Manager...")
        await app.state.tool_manager.stop_client()
    else:
        logger.info("MCP Tool Manager was not running or not initialized")
    logger.info("Application shutdown complete")


app = FastAPI(title="MultiAgent Medical Assistant", lifespan=lifespan)


@app.get("/health")
async def health_check(request: Request):
    """basic health check endpoint"""
    tool_manager_status = "stopped"
    graph_status = "not loaded"

    if hasattr(request.app.state, "tool_manager"):
        tool_manager_status = (
            "running" if request.app.state.tool_manager.is_running else "stopped"
        )
    if (
        hasattr(request.app.state, "main_graph")
        and request.app.state.main_graph is not None
    ):
        graph_status = "loaded"

    return {
        "status": "ok",
        "mcp_client": tool_manager_status,
        "main_graph": graph_status,
    }


from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    """Receives user query, invoke the main orchestrator graph, returns response"""

    if not hasattr(request.app.state, "tool_manager") or not hasattr(
        request.app.state, "main_graph"
    ):
        logger.error("tool manager or main graph not initialized in app state")
        raise HTTPException(status_code=500, detail="system not fully initialized")

    tool_manager: MCPToolManager = request.app.state.tool_manager
    graph = request.app.state.main_graph

    if graph is None:
        logger.error("Main graph failed to build during startup")
        raise HTTPException(status_code=500, detail="Agent orchestrator is unavailable")

    user_role = request.headers.get("X-User-Role", "patient")
    logger.info(f"Received chat request with simulated role {user_role}")

    session_id = (
        chat_request.session_id
        or request.cookies.get("session_id")
        or str(uuid.uuid4())
    )
    logger.info(f"using session ID (thread id): {session_id}")

    initial_state = AgentState(
        messages=[HumanMessage(content=chat_request.query)],
        current_input=chat_request.query,
        user_role=user_role,
    )

    config = {"configurable": {"thread_id": session_id}}

    try:
        logger.info(f"Invoking main graph for session {session_id}")
        final_state = await graph.ainvoke(initial_state, config=config)
        logger.info(f"Graph invocation complete for session {session_id}")

        output = final_state.get("final_output")
        if output is None:
            if final_state.get("messages"):
                last_message = final_state["messages"][-1]
                if isinstance(last_message, HumanMessage):
                    logger.warning("graph ended on a HumanMessage. Returning error")
                    output = "error: could not determine final response"
                else:
                    output = last_message.content
            else:
                logger.error(
                    f"Graph execution finished for session {session_id} but no 'final_output' or messages found in state"
                )
                output = "Sorry, an internal error occurred"

        response_content = {"response": output, "session_id": session_id}
        response = JSONResponse(content=response_content)
        response.set_cookie(key="session_id", value=session_id, httponly=True)

        return response

    except Exception as e:
        logger.error(
            f"Error during graph invocation for session {session_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error processing request {e}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Running uvicorn directly...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
