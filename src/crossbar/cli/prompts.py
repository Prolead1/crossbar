"""Interactive prompting with immediate validation."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

__all__ = ["_positive", "_non_negative", "_prompt"]


def _positive(label: str) -> Callable[[object], None]:
    """Return a validator that requires a strictly positive value."""

    def _check(value) -> None:
        if not value > 0:
            raise ValueError(f"{label} must be strictly positive")

    return _check


def _non_negative(label: str) -> Callable[[object], None]:
    """Return a validator that requires a non-negative value."""

    def _check(value) -> None:
        if value < 0:
            raise ValueError(f"{label} must be non-negative")

    return _check


def _prompt(
    label: str,
    default,
    cast: Callable[[str], object],
    choices: Optional[Sequence[str]] = None,
    allow_blank: bool = False,
    validate: Optional[Callable[[object], None]] = None,
    labels: Optional[dict] = None,
):
    """Ask for one value and fail immediately when it is invalid.

    A blank answer takes ``default`` (or ``None`` when ``allow_blank``);
    ``choices`` constrains string answers and ``validate`` applies a
    domain check.  ``labels`` maps each accepted choice to the longer
    name shown in parentheses, and either form is accepted as input.
    Invalid input raises :class:`ValueError` rather than re-prompting, so
    a mistake aborts the run.
    """

    def _shown(choice):
        return labels.get(choice, choice) if labels else choice

    hint = ""
    if choices is not None:
        hint += " (" + "/".join(_shown(c) for c in choices) + ")"
    if default is not None:
        hint += f" [{default}]"

    raw = input(f"{label}{hint}: ").strip()
    if not raw:
        if allow_blank:
            return None
        if default is None:
            raise ValueError(f"{label}: a value is required")
        value = default
    elif choices is not None:
        if raw in choices:
            chosen = raw
        else:
            matches = [c for c in choices if _shown(c) == raw]
            if not matches:
                raise ValueError(
                    f"{label}: choose one of "
                    + ", ".join(_shown(c) for c in choices)
                )
            chosen = matches[0]
        value = cast(chosen)
    else:
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: could not parse {raw!r}") from None

    if validate is not None:
        validate(value)
    return value
