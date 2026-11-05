"""Bridge from the workflow stages to model_tools.

Every stage calls ``client.call_model`` (through the module, not a copied name), so tests and
callers can replace the model in one place.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..data.io import WorkflowError
from ..model_tools import InvalidModelJSON, ModelToolError, WrongQuestionCount, run as run_model


class InvalidGeneratorJSON(WorkflowError):
    """The generator did not return valid JSON; asking again may succeed."""


class IncompleteBatch(WorkflowError):
    """The generator returned the wrong number of questions; a smaller batch may succeed."""


def call_model(config: dict[str, Any], command: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return run_model(argparse.Namespace(command=command, **kwargs), config)
    except InvalidModelJSON as exc:
        error_type = InvalidGeneratorJSON if exc.role == "generator" else WorkflowError
        raise error_type(f"{command}: {exc}") from exc
    except WrongQuestionCount as exc:
        raise IncompleteBatch(f"{command}: {exc}") from exc
    except ModelToolError as exc:
        raise WorkflowError(f"{command}: {exc}") from exc
