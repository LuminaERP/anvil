"""
Tool registry. Every tool is:
  1. A Python callable taking kwargs, returning str (the observation).
  2. An OpenAI-compatible tool schema dict for function calling.
  3. A classification ("read" | "write" | "shell") for safety routing.

Tools raise ToolError on failure; the executor turns that into an error
observation and continues (the agent should decide to retry or abandon).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


class ToolError(Exception):
    """Recoverable tool failure. Message is shown back to the model."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON schema, OpenAI style
    category: str                       # "read" | "write" | "shell"
    fn: Callable[..., str]              # kwargs -> observation string

    def schema(self) -> dict:
        """OpenAI tools[] entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name!r}. available: {list(self._tools)}")
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


REGISTRY = ToolRegistry()


def register(tool: Tool) -> Tool:
    REGISTRY.register(tool)
    return tool
