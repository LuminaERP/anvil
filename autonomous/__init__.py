"""Anvil — autonomous coding agent.

On import, we initialise the OpenTelemetry tracer (no-op if no OTLP endpoint
is configured in env). Every subsequent span from planner / executor /
reflector / tool-call flows through whichever exporter is configured.

This module stays tiny — real entry points live in main.py (one-shot CLI)
and daemon.py (long-running backlog processor).
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# Eager telemetry initialisation so the first agent turn already has tracing.
# init() is a no-op if OTEL_EXPORTER_OTLP_ENDPOINT is unset, so this is safe.
try:
    from . import telemetry as _telemetry
    _telemetry.init(service_name=os.environ.get("OTEL_SERVICE_NAME", "anvil"))
except Exception as exc:
    _log.debug("telemetry init skipped: %s", exc)
