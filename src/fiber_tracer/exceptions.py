"""Custom exceptions for fiber_tracer."""


class FiberTracerError(Exception):
    """Base exception."""


class ConfigError(FiberTracerError):
    """Invalid configuration."""


class DataError(FiberTracerError):
    """Data loading or validation error."""


class BackendNotAvailableError(FiberTracerError):
    """Optional backend dependency is missing."""


class ValidationError(FiberTracerError):
    """Validation metric or phantom generation failed."""
