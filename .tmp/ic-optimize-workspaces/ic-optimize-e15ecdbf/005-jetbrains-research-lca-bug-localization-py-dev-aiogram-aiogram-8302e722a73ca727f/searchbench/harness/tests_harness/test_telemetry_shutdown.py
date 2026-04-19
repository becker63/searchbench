from __future__ import annotations

import logging

from harness.telemetry import tracing


def test_otel_shutdown_timeout_noise_is_filtered(caplog):
    logger = logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        logger.error("Failed to export span batch due to timeout, max retries or shutdown.")
        logger.error("real telemetry failure")

    assert "Failed to export span batch due to timeout" not in caplog.text
    assert "real telemetry failure" in caplog.text
    assert tracing is not None

