"""
Command whitelist/denylist + write-path guard.
Called before every shell execution and every filesystem write.
"""
from __future__ import annotations
from pathlib import Path
from ..config import CONFIG


def classify_shell(cmd: str) -> tuple[str, str]:
    """
    Return (verdict, reason).
    verdict in {"allow", "deny", "review"}:
      allow  - matches allowlist, no denylist hit
      deny   - hit denylist (never runs without explicit --yolo override)
      review - doesn't match allowlist; human must confirm
    """
    safety = CONFIG["safety"]
    s = cmd.strip()

    # Denylist takes precedence
    for pat in safety.shell_denylist:
        if pat in s:
            return "deny", f"denylist match: {pat!r}"

    # Allowlist: prefix match
    for pref in safety.shell_allowlist:
        if s == pref or s.startswith(pref + " ") or s.startswith(pref + "\t"):
            return "allow", f"allowlist: {pref!r}"

    return "review", "not in allowlist"


def classify_write(path: str) -> tuple[str, str]:
    """Writes to /, /etc, /usr, /root/.ssh are never auto-allowed."""
    p = Path(path).resolve()
    forbidden_prefixes = ("/etc", "/usr", "/bin", "/sbin", "/boot", "/root/.ssh", "/var/lib")
    for bad in forbidden_prefixes:
        if str(p).startswith(bad):
            return "deny", f"forbidden path prefix: {bad}"

    workspace = CONFIG["paths"].workspace.resolve()
    if not str(p).startswith(str(workspace)):
        return "review", f"outside workspace {workspace}"

    if CONFIG["safety"].require_approval_for_writes:
        return "review", "writes require approval"
    return "allow", "inside workspace"
