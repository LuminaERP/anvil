"""Importing this package registers all tools in the REGISTRY."""
import logging
import os
import shutil

from . import filesystem  # noqa: F401
from . import shell       # noqa: F401
from . import research    # noqa: F401
from . import git_tool    # noqa: F401
from . import code_tool   # noqa: F401
from . import context7    # noqa: F401
from . import nia         # noqa: F401
from .base import REGISTRY, Tool, ToolError, register  # noqa: F401

_log = logging.getLogger(__name__)

# pskit is opt-in (pre-warming pwsh sessions + spinning an MCP subprocess adds
# a couple of seconds to startup). Enable with PSKIT_ENABLED=1 or by calling
# register_pskit() from application startup code.
_PSKIT_ON_PATH = shutil.which("pskit") is not None

if os.environ.get("PSKIT_ENABLED") == "1":
    try:
        from . import pskit_mcp
        pskit_mcp.register_pskit_tools()
    except Exception as exc:  # never break the agent over a pskit hiccup
        _log.warning("pskit auto-register failed: %s", exc)
elif _PSKIT_ON_PATH:
    _log.info(
        "pskit detected on PATH but PSKIT_ENABLED is not set. Set "
        "PSKIT_ENABLED=1 (or call autonomous.tools.register_pskit()) "
        "to gain 33 additional tools (disk/port/process/gpu, git extras, "
        "http_request, install_package) behind pskit's neural safety pipeline."
    )


def register_pskit() -> int:
    """Explicit opt-in: register pskit's MCP tools into REGISTRY.

    Returns the number of tools registered (0 if pskit unavailable).
    Safe to call multiple times — becomes a no-op after first success.
    """
    try:
        from . import pskit_mcp
        return pskit_mcp.register_pskit_tools()
    except Exception as exc:
        _log.warning("register_pskit failed: %s", exc)
        return 0


def pskit_available() -> bool:
    """True if `pskit` is on PATH — caller may choose to auto-enable."""
    return _PSKIT_ON_PATH
