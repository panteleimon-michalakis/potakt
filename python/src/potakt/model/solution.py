"""Solutions: which resource works which task, for one version of a Decision Model."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from potakt.model.common import Loc, StrictModel, raise_errors


class _FrozenModel(StrictModel):
    model_config = ConfigDict(frozen=True)


class Assignment(_FrozenModel):
    """Resource ``resource_id`` works task ``task_id``."""

    task_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)


class SolutionMetadata(_FrozenModel):
    """Optional, free-form information about where a solution came from."""

    produced_by: str | None = Field(default=None, min_length=1)
    created_at: str | None = Field(default=None, min_length=1)
    notes: str | None = Field(default=None, min_length=1)


class Solution(_FrozenModel):
    """A set of assignments for model ``model_id`` at ``model_version``.

    Tasks may be left unassigned (``coverage`` scores them); a pair may appear only once.
    """

    schema_version: Literal["0.1"]
    model_id: str = Field(min_length=1)
    model_version: int = Field(ge=1)
    assignments: tuple[Assignment, ...]
    metadata: SolutionMetadata | None = None

    @model_validator(mode="after")
    def _check_unique_pairs(self) -> Self:
        first_index: dict[Assignment, int] = {}
        errors: list[tuple[Loc, str]] = []
        for index, assignment in enumerate(self.assignments):
            if assignment in first_index:
                message = (
                    f"duplicate assignment of resource {assignment.resource_id!r} to task "
                    f"{assignment.task_id!r} (first at /assignments/{first_index[assignment]})"
                )
                errors.append((("assignments", index), message))
            else:
                first_index[assignment] = index
        if errors:
            raise_errors("Solution", errors)
        return self
