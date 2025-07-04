from pydantic import BaseModel, Field
from typing import Optional, Any, Iterable


class BaseIssue(BaseModel):
    id: int
    data: str
    ref_id: Optional[int] = None
    ref_data: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        id: int,
        data: str,
        ref_id: Optional[int] = None,
        ref_data: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> "BaseIssue":
        """Creates a BaseIssue instance from individual arguments."""
        return cls(
            id=id,
            data=data,
            ref_id=ref_id,
            ref_data=ref_data,
            meta=meta or {},
        )

    @classmethod
    def from_batch(
        cls,
        data_batch: Iterable[
            tuple[int, str, Optional[int], Optional[str], Optional[dict[str, Any]]]
        ],
    ) -> list["BaseIssue"]:
        """Creates a list of BaseIssue instances from a batch of tuples."""
        return [cls.from_data(*item) for item in data_batch]
