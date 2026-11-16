"""Stage transition rules and default follow-up rules. Plain data, easy to override."""
from __future__ import annotations

from dataclasses import dataclass

from careerops.domain import Stage

TERMINAL = frozenset({Stage.REJECTED, Stage.WITHDRAWN})

# Allowed moves from each stage. Terminal stages have none; use reopen().
TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.SAVED: frozenset({Stage.PREPARING, Stage.APPLIED, Stage.REJECTED, Stage.WITHDRAWN}),
    Stage.PREPARING: frozenset({Stage.SAVED, Stage.APPLIED, Stage.REJECTED, Stage.WITHDRAWN}),
    Stage.APPLIED: frozenset(
        {Stage.SCREEN, Stage.INTERVIEW, Stage.OFFER, Stage.REJECTED, Stage.WITHDRAWN}
    ),
    Stage.SCREEN: frozenset({Stage.INTERVIEW, Stage.OFFER, Stage.REJECTED, Stage.WITHDRAWN}),
    Stage.INTERVIEW: frozenset({Stage.OFFER, Stage.REJECTED, Stage.WITHDRAWN}),
    Stage.OFFER: frozenset({Stage.REJECTED, Stage.WITHDRAWN}),
    Stage.REJECTED: frozenset(),
    Stage.WITHDRAWN: frozenset(),
}

# Stages an application can be reopened into.
REOPEN_TARGETS = frozenset(
    {Stage.SAVED, Stage.PREPARING, Stage.APPLIED, Stage.SCREEN, Stage.INTERVIEW}
)


@dataclass(frozen=True)
class FollowUpRule:
    """One follow-up to schedule.

    key: unique name for the rule. Used so a rule is scheduled only once per application.
    days: days to wait.
    note: text of the follow-up.
    after: None means days after the application entered the stage.
        Otherwise the key of another rule. This rule is scheduled when that
        rule's follow-up is completed, and counts days from the completion date.
    """

    key: str
    days: int
    note: str
    after: str | None = None


DEFAULT_RULES: dict[Stage, list[FollowUpRule]] = {
    Stage.APPLIED: [
        FollowUpRule("applied_1", 7, "No response yet. Send a follow-up."),
        FollowUpRule("applied_2", 7, "Second follow-up.", after="applied_1"),
        FollowUpRule("applied_3", 14, "Last follow-up, then move on.", after="applied_2"),
    ],
    Stage.SCREEN: [
        FollowUpRule("screen_thanks", 1, "Send a thank-you after the screen."),
    ],
    Stage.INTERVIEW: [
        FollowUpRule("interview_thanks", 3, "Send a thank-you note."),
        FollowUpRule("interview_check", 7, "Check on next steps.", after="interview_thanks"),
    ],
    Stage.OFFER: [
        FollowUpRule("offer_reply", 5, "Reply to the offer."),
    ],
}
