"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import NewMatch from "@/components/NewMatch";
import { StatusBadge, formatTime } from "@/components/common";
import { Api, type Match } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    new Api()
      .listMatches()
      .then(setMatches)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <h1>Matches</h1>
      <NewMatch onUploaded={(id) => router.push(`/match/?id=${id}`)} />

      <section className="card">
        {error && (
          <p className="error">
            Can&apos;t reach the API ({error}). Check it is running and the address in{" "}
            <Link href="/settings/">Settings</Link>.
          </p>
        )}
        {!error && matches === null && <p className="muted">Loading…</p>}
        {matches?.length === 0 && <p className="muted">No matches yet. Upload one above.</p>}
        {matches && matches.length > 0 && (
          <ul className="list">
            {matches.map((m) => (
              <li key={m.id}>
                <Link className="item" href={`/match/?id=${m.id}`}>
                  <span className="grow">
                    <strong>{m.title}</strong>
                    <br />
                    <span className="muted small">
                      {new Date(m.created_at).toLocaleString()}
                      {m.duration_s ? ` · ${formatTime(m.duration_s)}` : ""}
                    </span>
                  </span>
                  <StatusBadge status={m.status} />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
