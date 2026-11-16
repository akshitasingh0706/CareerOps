"""Hard filters. Pure code, no LLM. Each returns a FilterResult with a specific detail."""
from __future__ import annotations

from careerops.domain import FilterResult, Job, Seniority, UserProfile

from .config import FilterConfig, MissingPolicy
from .text import contains_term

UNKNOWN_PREFIX = "unknown:"

# Position on the seniority ladder. Manager sits level with senior. UNKNOWN is absent.
SENIORITY_ORDER: dict[Seniority, int] = {
    Seniority.INTERN: 0,
    Seniority.JUNIOR: 1,
    Seniority.MID: 2,
    Seniority.SENIOR: 3,
    Seniority.MANAGER: 3,
    Seniority.STAFF: 4,
    Seniority.PRINCIPAL: 5,
}
_TOP = max(SENIORITY_ORDER.values())


def is_unknown(result: FilterResult) -> bool:
    return result.passed and result.detail.startswith(UNKNOWN_PREFIX)


def _missing(name: str, policy: MissingPolicy, what: str) -> FilterResult:
    if policy == "fail":
        return FilterResult(name=name, passed=False, detail=f"{what}; policy is to fail")
    if policy == "pass":
        return FilterResult(name=name, passed=True, detail=f"{what}; policy is to pass")
    return FilterResult(name=name, passed=True, detail=f"{UNKNOWN_PREFIX} {what}")


def _is_remote(job: Job) -> bool | None:
    if job.remote is not None:
        return job.remote
    if job.location and "remote" in job.location.lower():
        return True
    return None


def _location_match(job: Job, profile: UserProfile) -> str | None:
    if not job.location:
        return None
    loc = job.location.lower()
    for wanted in profile.locations:
        if wanted.strip() and wanted.strip().lower() in loc:
            return wanted
    return None


def location_filter(job: Job, profile: UserProfile, cfg: FilterConfig | None = None) -> FilterResult:
    """Pass if the job is remote and the user accepts remote, or the job location matches one of
    the user's locations. If the user lists no locations and does not require them, pass."""
    cfg = cfg or FilterConfig()
    remote = _is_remote(job)
    if remote and profile.remote_ok:
        return FilterResult(name="location", passed=True, detail="remote job and remote is ok")
    match = _location_match(job, profile)
    if match:
        return FilterResult(
            name="location", passed=True, detail=f"location '{job.location}' matches '{match}'"
        )
    if not profile.locations and profile.remote_ok:
        return FilterResult(name="location", passed=True, detail="no location preference set")
    if remote is None and not job.location:
        return _missing("location", cfg.location_missing, "job has no location and no remote flag")
    if remote:
        return FilterResult(
            name="location", passed=False, detail="remote job but user does not accept remote"
        )
    where = job.location or "not stated"
    wanted = ", ".join(profile.locations) or "none listed"
    return FilterResult(
        name="location", passed=False, detail=f"location '{where}' is not in [{wanted}]"
    )


def seniority_range(profile: UserProfile) -> tuple[int, int] | None:
    """The user's accepted seniority range as ladder positions, or None if unconstrained."""
    lo = SENIORITY_ORDER.get(profile.min_seniority)
    hi = SENIORITY_ORDER.get(profile.max_seniority)
    if lo is None and hi is None:
        return None
    return (0 if lo is None else lo, _TOP if hi is None else hi)


def seniority_filter(job: Job, profile: UserProfile, cfg: FilterConfig | None = None) -> FilterResult:
    """Pass if the job seniority is inside [min_seniority, max_seniority]."""
    cfg = cfg or FilterConfig()
    rng = seniority_range(profile)
    if rng is None:
        return FilterResult(name="seniority", passed=True, detail="no seniority preference set")
    pos = SENIORITY_ORDER.get(job.seniority)
    if pos is None:
        return _missing("seniority", cfg.seniority_missing, "job seniority is unknown")
    lo, hi = rng
    ok = lo <= pos <= hi
    rng_txt = f"{profile.min_seniority.value} to {profile.max_seniority.value}"
    verb = "is inside" if ok else "is outside"
    return FilterResult(
        name="seniority", passed=ok, detail=f"job level {job.seniority.value} {verb} {rng_txt}"
    )


def offered_salary(job: Job) -> int | None:
    """The best figure the posting shows: salary_max if present, else salary_min."""
    return job.salary_max if job.salary_max is not None else job.salary_min


def compensation_filter(
    job: Job, profile: UserProfile, cfg: FilterConfig | None = None
) -> FilterResult:
    """Pass if the top of the posted range reaches the user's minimum salary.
    Missing salary, or a currency different from the user's, is handled by the missing policy."""
    cfg = cfg or FilterConfig()
    if profile.min_salary is None:
        return FilterResult(name="compensation", passed=True, detail="no minimum salary set")
    offer = offered_salary(job)
    if offer is None:
        return _missing("compensation", cfg.compensation_missing, "job does not state a salary")
    if job.salary_currency and job.salary_currency != profile.salary_currency:
        return _missing(
            "compensation",
            cfg.compensation_missing,
            f"salary is in {job.salary_currency}, user minimum is in {profile.salary_currency}",
        )
    ok = offer >= profile.min_salary
    rel = "meets" if ok else "is below"
    return FilterResult(
        name="compensation",
        passed=ok,
        detail=f"top of range {offer} {rel} minimum {profile.min_salary}",
    )


def avoid_companies_filter(job: Job, profile: UserProfile) -> FilterResult:
    """Fail if the company name or id equals an avoided company, ignoring case."""
    avoided = {c.strip().lower() for c in profile.avoid_companies if c.strip()}
    hit = {job.company_name.strip().lower(), job.company_id.strip().lower()} & avoided
    if hit:
        return FilterResult(
            name="avoid_companies", passed=False, detail=f"'{job.company_name}' is on the avoid list"
        )
    return FilterResult(name="avoid_companies", passed=True, detail="company is not on the avoid list")


def must_have_skills_filter(job: Job, profile: UserProfile) -> FilterResult:
    """Fail if any must-have skill is absent from the title and description (whole-word match)."""
    if not profile.must_have_skills:
        return FilterResult(name="must_have_skills", passed=True, detail="no must-have skills set")
    text = f"{job.title}\n{job.description}"
    missing = [s for s in profile.must_have_skills if not contains_term(text, s)]
    if missing:
        return FilterResult(
            name="must_have_skills", passed=False, detail="missing from description: " + ", ".join(missing)
        )
    return FilterResult(
        name="must_have_skills",
        passed=True,
        detail="all present: " + ", ".join(profile.must_have_skills),
    )


def apply_filters(
    job: Job, profile: UserProfile, cfg: FilterConfig | None = None
) -> list[FilterResult]:
    """Run every hard filter. A job is excluded if any result has passed=False."""
    return [
        location_filter(job, profile, cfg),
        seniority_filter(job, profile, cfg),
        compensation_filter(job, profile, cfg),
        avoid_companies_filter(job, profile),
        must_have_skills_filter(job, profile),
    ]


def passes_all(results: list[FilterResult]) -> bool:
    return all(r.passed for r in results)
