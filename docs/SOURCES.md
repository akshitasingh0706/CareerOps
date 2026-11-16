# Job sources and compliance

CareerOps must not put the user's accounts at risk. This file sets out where job data comes from.

## Allowed sources

| Source | How | Notes |
|---|---|---|
| Greenhouse, Lever, Ashby | Public job board JSON APIs | Built for third-party reads. No auth. |
| Company career pages | Fetch with robots.txt respected, JSON-LD `JobPosting` where present | Polite rate limits. |
| Alert emails (LinkedIn, Wellfound, Indeed, others) | Parse emails the user receives, read through their own Gmail | The user's own mail. No scraping. |
| Licensed aggregator APIs | Adzuna, JSearch, TheirStack, Fantastic.jobs and similar | Provider holds the data agreement. Optional and may cost money. |
| Browser capture | A button or bookmarklet the user presses on a job they are viewing | The user is browsing. No bot. |
| Pasted URL or text | Manual | Fallback for anything else. |

## Not allowed

- Logging in to LinkedIn or Wellfound with automation, or scraping them.
- Automated clicking, applying, or messaging on those sites.
- Anything that needs the user's password or session cookie.

## Why

- LinkedIn's User Agreement bans scrapers and bots. LinkedIn restricts accounts and has sued scrapers. LinkedIn has no job search API for individuals.
- Wellfound's terms ban scraping and it uses bot protection.
- The risk lands on the user's account, so CareerOps stays out of it.

## Source tagging

Every job records its `Source` and how it was obtained. Jobs from aggregators and alert emails
keep the original listing URL so the user can apply on the real site.
