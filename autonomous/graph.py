"""
LangGraph assembly.

States & transitions:

   START
     │
     ▼
   plan ──(status="failed")──► END
     │
     ▼
  execute ─(more subgoals)─► execute
     │
     ▼  (all done or reflection triggered)
  reflect ──(done)──► END
     │
     ├──(stuck)──► plan  (replan)
     │
     └──(continue, still pending)──► execute

Budget enforcement: max_iterations cap is checked at reflect; if we've
looped too much, we force "done" with a "budget exhausted" note.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .config import CONFIG
from .nodes import plan_node, executor_node, reflector_node


def _route_after_plan(state: AgentState) -> Literal["execute", "end"]:
    if state.get("status") == "failed":
        return "end"
    return "execute"


def _route_after_execute(state: AgentState) -> Literal["execute", "reflect"]:
    # After each subgoal, go to reflect. Reflect decides whether to return to execute.
    return "reflect"


def _route_after_reflect(state: AgentState) -> Literal["execute", "plan", "end"]:
    status = state.get("status", "executing")
    iterations = state.get("iterations", 0)
    if status == "done" or iterations >= CONFIG["budget"].max_iterations:
        return "end"
    if status == "replanning":
        return "plan"
    return "execute"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", executor_node)
    g.add_node("reflect", reflector_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _route_after_plan, {"execute": "execute", "end": END})
    g.add_conditional_edges("execute", _route_after_execute, {"reflect": "reflect", "execute": "execute"})
    g.add_conditional_edges("reflect", _route_after_reflect, {"execute": "execute", "plan": "plan", "end": END})
    return g


def compile_graph_with_checkpointer():
    """Compile the graph with a SqliteSaver checkpointer so runs are resumable."""
    ckpt_path = CONFIG["paths"].checkpoint_db
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ckpt_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph().compile(checkpointer=saver)
