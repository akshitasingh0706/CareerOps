"use client";

import { useCallback, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type RankedJob = {
  id: string;
  title: string;
  company: string;
  location: string | null;
  remote: boolean | null;
  url: string;
  source: string;
  seniority: string;
  salary_min: number | null;
  salary_max: number | null;
  score: number;
  components: Record<string, number>;
  unknown_filters: string[];
  failed_filters: string[];
};

type Ranked = {
  ranked: RankedJob[];
  filtered_out: RankedJob[];
  personalization: { status: string; needed?: string };
};

type Profile = {
  target_titles: string[];
  locations: string[];
  remote_ok: boolean;
  min_salary: number | null;
  must_have_skills: string[];
  avoid_companies: string[];
  interests: string[];
  min_seniority: string;
  max_seniority: string;
  salary_currency: string;
};

type AppRow = {
  application: { id: string; stage: string };
  job: { title: string; company_name: string; url: string } | null;
};

const SENIORITY = [
  "unknown", "intern", "junior", "mid", "senior", "staff", "principal", "manager",
];
const STAGES = ["saved", "preparing", "applied", "screen", "interview", "offer", "rejected", "withdrawn"];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

const csv = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

export default function Home() {
  const [ranked, setRanked] = useState<Ranked | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [apps, setApps] = useState<AppRow[]>([]);
  const [pipeline, setPipeline] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [ats, setAts] = useState("greenhouse");
  const [token, setToken] = useState("");

  const load = useCallback(async () => {
    try {
      const [r, p, a, pl] = await Promise.all([
        api<Ranked>("/jobs/ranked?limit=40"),
        api<Profile>("/profile"),
        api<AppRow[]>("/applications"),
        api<Record<string, number>>("/pipeline"),
      ]);
      setRanked(r);
      setProfile(p);
      setApps(a);
      setPipeline(pl);
      setError(null);
    } catch (e) {
      setError(
        `Cannot reach the API at ${API}. Is it running? (${e instanceof Error ? e.message : e})`,
      );
    }
  }, []);

  useEffect(() => {
    // Initial data fetch from the API on mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto max-w-6xl p-6 text-sm text-zinc-900 dark:text-zinc-100">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">CareerOps</h1>
        <span className="text-zinc-500">
          Ranking: {ranked?.personalization.status ?? "loading"}
        </span>
      </header>

      {error && (
        <p className="mb-4 rounded border border-red-300 bg-red-50 p-3 text-red-800 dark:bg-red-950 dark:text-red-200">
          {error}
        </p>
      )}

      <section className="mb-6 grid gap-4 md:grid-cols-2">
        <Card title="Watch a company">
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (!token.trim()) return;
              void run(async () => {
                await api("/companies", {
                  method: "POST",
                  body: JSON.stringify({ ats, token: token.trim() }),
                });
                setToken("");
              });
            }}
          >
            <select className="field" value={ats} onChange={(e) => setAts(e.target.value)}>
              <option>greenhouse</option>
              <option>lever</option>
              <option>ashby</option>
            </select>
            <input
              className="field flex-1"
              placeholder="board token, e.g. anthropic"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <button className="btn" disabled={busy}>Watch</button>
          </form>
          <button
            className="btn mt-3"
            disabled={busy}
            onClick={() => void run(() => api("/poll", { method: "POST" }))}
          >
            {busy ? "Working..." : "Fetch new jobs"}
          </button>
        </Card>

        <Card title="Pipeline">
          <div className="flex flex-wrap gap-2">
            {STAGES.map((s) => (
              <span key={s} className="rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800">
                {s}: <b>{pipeline[s] ?? 0}</b>
              </span>
            ))}
          </div>
        </Card>
      </section>

      {profile && (
        <ProfileForm
          profile={profile}
          busy={busy}
          onSave={(p) => run(() => api("/profile", { method: "PUT", body: JSON.stringify(p) }))}
        />
      )}

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-medium">
          Ranked jobs {ranked && `(${ranked.ranked.length} shown)`}
        </h2>
        {ranked?.ranked.length === 0 && (
          <p className="text-zinc-500">No jobs yet. Watch a company, then fetch new jobs.</p>
        )}
        <ul className="space-y-2">
          {ranked?.ranked.map((j) => (
            <li key={j.id} className="rounded border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <a href={j.url} target="_blank" rel="noreferrer" className="font-medium underline">
                    {j.title}
                  </a>
                  <div className="text-zinc-500">
                    {j.company} · {j.location ?? "location unknown"} · {j.seniority}
                    {j.salary_min ? ` · $${j.salary_min.toLocaleString()}+` : ""}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-zinc-500">
                    {Object.entries(j.components).map(([k, v]) => (
                      <span key={k}>{k} {v.toFixed(2)}</span>
                    ))}
                    {j.unknown_filters.length > 0 && (
                      <span className="text-amber-600">check: {j.unknown_filters.join(", ")}</span>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="rounded bg-zinc-100 px-2 py-1 font-mono dark:bg-zinc-800">
                    {j.score.toFixed(2)}
                  </span>
                  <button
                    className="btn"
                    disabled={busy}
                    onClick={() => void run(() => api(`/jobs/${j.id}/save`, { method: "POST" }))}
                  >
                    Save
                  </button>
                  <button
                    className="btn-quiet"
                    disabled={busy}
                    onClick={() => void run(() => api(`/jobs/${j.id}/skip`, { method: "POST", body: "{}" }))}
                  >
                    Skip
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-medium">Applications</h2>
        {apps.length === 0 && <p className="text-zinc-500">Nothing saved yet.</p>}
        <ul className="space-y-2">
          {apps.map((a) => (
            <li
              key={a.application.id}
              className="flex items-center justify-between rounded border border-zinc-200 p-3 dark:border-zinc-800"
            >
              <span>
                {a.job?.title ?? "Unknown job"}
                <span className="text-zinc-500"> · {a.job?.company_name}</span>
              </span>
              <select
                className="field"
                value={a.application.stage}
                disabled={busy}
                onChange={(e) =>
                  void run(() =>
                    api(`/applications/${a.application.id}/advance`, {
                      method: "POST",
                      body: JSON.stringify({ stage: e.target.value }),
                    }),
                  )
                }
              >
                {STAGES.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded border border-zinc-200 p-4 dark:border-zinc-800">
      <h2 className="mb-3 font-medium">{title}</h2>
      {children}
    </div>
  );
}

function ProfileForm({
  profile,
  busy,
  onSave,
}: {
  profile: Profile;
  busy: boolean;
  onSave: (p: Profile) => void;
}) {
  const [titles, setTitles] = useState(profile.target_titles.join(", "));
  const [locations, setLocations] = useState(profile.locations.join(", "));
  const [skills, setSkills] = useState(profile.must_have_skills.join(", "));
  const [interests, setInterests] = useState(profile.interests.join(", "));
  const [avoid, setAvoid] = useState(profile.avoid_companies.join(", "));
  const [remote, setRemote] = useState(profile.remote_ok);
  const [minSalary, setMinSalary] = useState(profile.min_salary?.toString() ?? "");
  const [minSen, setMinSen] = useState(profile.min_seniority);
  const [maxSen, setMaxSen] = useState(profile.max_seniority);

  return (
    <details className="mb-6 rounded border border-zinc-200 p-4 dark:border-zinc-800" open={!profile.target_titles.length}>
      <summary className="cursor-pointer font-medium">What you want (drives the ranking)</summary>
      <form
        className="mt-3 grid gap-3 md:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          onSave({
            ...profile,
            target_titles: csv(titles),
            locations: csv(locations),
            must_have_skills: csv(skills),
            interests: csv(interests),
            avoid_companies: csv(avoid),
            remote_ok: remote,
            min_salary: minSalary ? Number(minSalary) : null,
            min_seniority: minSen,
            max_seniority: maxSen,
          });
        }}
      >
        <Label text="Target titles (comma separated)">
          <input className="field w-full" value={titles} onChange={(e) => setTitles(e.target.value)} />
        </Label>
        <Label text="Locations">
          <input className="field w-full" value={locations} onChange={(e) => setLocations(e.target.value)} />
        </Label>
        <Label text="Must-have skills">
          <input className="field w-full" value={skills} onChange={(e) => setSkills(e.target.value)} />
        </Label>
        <Label text="Interests (used for growth fit)">
          <input className="field w-full" value={interests} onChange={(e) => setInterests(e.target.value)} />
        </Label>
        <Label text="Avoid companies">
          <input className="field w-full" value={avoid} onChange={(e) => setAvoid(e.target.value)} />
        </Label>
        <Label text="Minimum salary (USD)">
          <input className="field w-full" inputMode="numeric" value={minSalary} onChange={(e) => setMinSalary(e.target.value)} />
        </Label>
        <Label text="Seniority from">
          <select className="field w-full" value={minSen} onChange={(e) => setMinSen(e.target.value)}>
            {SENIORITY.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Label>
        <Label text="Seniority to">
          <select className="field w-full" value={maxSen} onChange={(e) => setMaxSen(e.target.value)}>
            {SENIORITY.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={remote} onChange={(e) => setRemote(e.target.checked)} />
          Remote is fine
        </label>
        <div className="md:col-span-2">
          <button className="btn" disabled={busy}>Save profile and re-rank</button>
        </div>
      </form>
    </details>
  );
}

function Label({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-zinc-500">{text}</span>
      {children}
    </label>
  );
}
