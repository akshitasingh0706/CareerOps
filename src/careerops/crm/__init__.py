"""Application CRM: applications, stages, submissions, contacts, follow-ups."""
from careerops.crm.rules import DEFAULT_RULES, TRANSITIONS, FollowUpRule
from careerops.crm.service import (
    CRM,
    CRMError,
    DueItem,
    IllegalTransition,
    NotFound,
    Submission,
    TimelineEvent,
)

__all__ = [
    "CRM",
    "DEFAULT_RULES",
    "TRANSITIONS",
    "CRMError",
    "DueItem",
    "FollowUpRule",
    "IllegalTransition",
    "NotFound",
    "Submission",
    "TimelineEvent",
]
