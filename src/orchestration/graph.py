"""Connect the Supervisor and three agents with LangGraph."""

from langgraph.graph import END, START, StateGraph

from src.orchestration.state import State, ResearchResult, DataResult, SummaryResult
from src.orchestration.supervisor import Supervisor


def build_graph(supervisor=None, research_agent=None, data_agent=None, summary_agent=None):
    # Use supplied debug wrappers or the real agents.
    if supervisor is None:
        supervisor = Supervisor()
    if research_agent is None:
        from src.agents.research import research
        research_agent = research
    if data_agent is None:
        from src.agents.data import data
        data_agent = data
    if summary_agent is None:
        from src.agents.summary import summary
        summary_agent = summary

    async def supervisor_node(state: State):
        if state.next_agent == "end":
            return {}
        return await supervisor(state.model_copy(deep=True))

    # Each node updates only its own result.
    async def research_node(state: State):
        try:
            output = await research_agent(state.model_copy(deep=True))
            result = ResearchResult.model_validate(output)
        except Exception:
            result = ResearchResult(status="failed", error="Research execution failed.")
        return {"research_result": result}

    async def data_node(state: State):
        try:
            output = await data_agent(state.model_copy(deep=True))
            result = DataResult.model_validate(output)
        except Exception:
            result = DataResult(status="failed", explanation="", error="Data execution failed.")
        return {"data_result": result}

    async def summary_node(state: State):
        try:
            output = await summary_agent(state.model_copy(deep=True))
            result = SummaryResult.model_validate(output)
        except Exception:
            result = SummaryResult(status="failed", error="Summary execution failed.")
        return {"summary_result": result}

    def route(state: State):
        return state.next_agent

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("data", data_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {
        "research": "research",
        "data": "data",
        "summary": "summary",
        "end": END,
    })
    graph.add_edge("research", "supervisor")
    graph.add_edge("data", "supervisor")
    graph.add_edge("summary", "supervisor")
    return graph.compile()


async def run_workflow(graph, state):
    """Run one question using a graph created and reused by the caller."""
    result = await graph.ainvoke(state.model_dump())
    # Convert the returned dictionary to State for existing callers.
    return State.model_validate(result)
