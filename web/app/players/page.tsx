"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { PlayerCard } from "@/components/Report";
import { Api, type Person, type PlayerInMatch } from "@/lib/api";

// One page for everyone named in a match, and for one person: /players/?name=Maya
export default function PlayersPage() {
  return (
    <Suspense fallback={<p className="muted">Loading…</p>}>
      <People />
    </Suspense>
  );
}

function People() {
  const name = useSearchParams().get("name");
  const router = useRouter();
  const api = useMemo(() => new Api(), []);
  const [people, setPeople] = useState<Person[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.people().then(setPeople, (e: Error) => setError(e.message));
  }, [api]);

  if (name) return <PersonView api={api} name={name} />;
  return (
    <>
      <h1>Players</h1>
      <section className="card">
        {error && <p className="error">{error}</p>}
        {!error && people === null && <p className="muted">Loading…</p>}
        {people?.length === 0 && (
          <p className="muted" style={{ margin: 0 }}>
            Nobody is named yet. Open an analysed match and use &quot;Name the players&quot; in its report; each name then
            gets a page here with their highlights from every match.
          </p>
        )}
        {people && people.length > 0 && (
          <ul className="list">
            {people.map((p) => (
              <li key={p.name}>
                <button className="link item" onClick={() => router.push(`/players/?name=${encodeURIComponent(p.name)}`)}>
                  <span className="grow">
                    <strong>{p.name}</strong>
                  </span>
                  <span className="muted small">
                    {p.matches} match{p.matches === 1 ? "" : "es"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function PersonView({ api, name }: { api: Api; name: string }) {
  const [entries, setEntries] = useState<PlayerInMatch[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setEntries(null);
    api.person(name).then(setEntries, (e: Error) => setError(e.message));
  }, [api, name]);

  const totals = useMemo(() => {
    const done = (entries ?? []).filter((e) => e.report);
    const sum = (f: (e: PlayerInMatch) => number) => done.reduce((a, e) => a + f(e), 0);
    return {
      matches: entries?.length ?? 0,
      shots: sum((e) => e.report!.shots),
      winners: sum((e) => e.report!.winners),
      errors: sum((e) => e.report!.errors),
      net: done.length ? sum((e) => e.report!.zone_share.net ?? 0) / done.length : 0,
    };
  }, [entries]);

  return (
    <>
      <p className="small">
        <Link href="/players/">← All players</Link>
      </p>
      <h1>{name}</h1>
      {error && <p className="error">{error}</p>}
      {!error && entries === null && <p className="muted">Loading…</p>}
      {entries && (
        <section className="card">
          <div className="stats">
            <div className="stat">
              <strong>{totals.matches}</strong>
              <span className="muted small">Matches</span>
            </div>
            <div className="stat">
              <strong>{totals.shots}</strong>
              <span className="muted small">Shots</span>
            </div>
            <div className="stat">
              <strong>{totals.winners}</strong>
              <span className="muted small">Winners</span>
            </div>
            <div className="stat">
              <strong>{totals.errors}</strong>
              <span className="muted small">Errors</span>
            </div>
            <div className="stat">
              <strong>{Math.round(totals.net * 100)}%</strong>
              <span className="muted small">At the net</span>
            </div>
          </div>
        </section>
      )}
      {entries?.map((e) => (
        <section className="card" key={`${e.match_id}-${e.player}`}>
          <div className="row" style={{ marginBottom: 8 }}>
            <h2 className="grow" style={{ margin: 0 }}>
              <Link href={`/match/?id=${e.match_id}`}>{e.match_title}</Link>
            </h2>
            <span className="muted small">
              {new Date(e.match_date).toLocaleDateString()} · {e.player}
              {e.team ? ` · team ${e.team}` : ""}
            </span>
          </div>
          {e.report ? (
            <PlayerCard
              api={api}
              matchId={e.match_id}
              p={e.report}
              name={name}
              names={{ [e.player]: name }}
              shorts={e.shorts ?? []}
            />
          ) : (
            <p className="muted small" style={{ margin: 0 }}>
              This match has no report yet. Calibrate the court and analyse it to get one.
            </p>
          )}
        </section>
      ))}
    </>
  );
}
