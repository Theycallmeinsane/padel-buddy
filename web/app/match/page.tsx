"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Calibrator from "@/components/Calibrator";
import MatchViewer from "@/components/MatchViewer";
import Report from "@/components/Report";
import Results from "@/components/Results";
import { ProgressBar, StatusBadge, formatBytes, formatTime } from "@/components/common";
import { Api, ApiError, type Match, type Results as ResultsData } from "@/lib/api";

// Static export can't pre-render one page per match, so the id is a query
// parameter: /match/?id=...
export default function MatchPage() {
  return (
    <Suspense fallback={<p className="muted">Loading…</p>}>
      <MatchView />
    </Suspense>
  );
}

const ACTIVE = new Set(["queued", "processing"]);

function MatchView() {
  const id = useSearchParams().get("id") ?? "";
  const router = useRouter();
  const api = useMemo(() => new Api(), []);
  const [match, setMatch] = useState<Match | null>(null);
  const [results, setResults] = useState<ResultsData | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [seekTo, setSeekTo] = useState<{ t: number; n: number } | null>(null);
  const names = useMemo(() => match?.player_names ?? {}, [match?.player_names]);

  const refresh = useCallback(async () => {
    try {
      const m = await api.getMatch(id);
      setMatch(m);
      if (m.status === "done") setResults(await api.results(id));
    } catch (e) {
      setError(e instanceof ApiError && e.status === 404 ? "This match doesn't exist." : (e as Error).message);
    }
  }, [api, id]);

  useEffect(() => {
    if (id) refresh();
  }, [id, refresh]);

  // Follow progress while the worker is busy.
  useEffect(() => {
    if (!match || !ACTIVE.has(match.status)) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [match, refresh]);

  async function analyze() {
    setError(null);
    try {
      await api.analyze(id);
      setResults(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function saveNames(next: Record<string, string>) {
    try {
      setMatch(await api.namePlayers(id, next));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove() {
    if (!confirm("Delete this match and all its clips?")) return;
    try {
      await api.deleteMatch(id);
      router.push("/");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!id) return <p>No match selected. <Link href="/">Back to matches</Link></p>;
  if (error && !match) return <p className="error">{error} <Link href="/">Back to matches</Link></p>;
  if (!match) return <p className="muted">Loading…</p>;

  const uploaded = !["created", "uploading"].includes(match.status);
  const busy = ACTIVE.has(match.status);
  const step = match.status === "done" ? 3 : match.calibrated || match.job ? 2 : 1;

  return (
    <>
      <div className="row" style={{ marginBottom: 8 }}>
        <h1 className="grow" style={{ margin: 0 }}>{match.title}</h1>
        <StatusBadge status={match.status} />
      </div>
      <p className="muted small" style={{ marginTop: 0 }}>
        {new Date(match.created_at).toLocaleString()}
        {match.duration_s ? ` · ${formatTime(match.duration_s)}` : ""}
      </p>

      {!uploaded && (
        <section className="card">
          <h2>Upload not finished</h2>
          <p>
            {formatBytes(match.video_received)}
            {match.video_size ? ` of ${formatBytes(match.video_size)}` : ""} received. Go back to{" "}
            <Link href="/">Matches</Link> and pick the same file again to resume.
          </p>
        </section>
      )}

      {uploaded && (
        <section className="card">
          <div className="steps">
            <span className={step === 1 ? "on" : ""}>1 · Court (optional)</span>
            <span className={step === 2 ? "on" : ""}>2 · Analyse</span>
            <span className={step === 3 ? "on" : ""}>3 · Watch</span>
          </div>
          {calibrating ? (
            <Calibrator
              api={api}
              matchId={id}
              durationS={match.duration_s}
              onSaved={() => {
                setCalibrating(false);
                refresh();
              }}
              onCancel={() => setCalibrating(false)}
            />
          ) : (
            <>
              <p style={{ marginTop: 0 }}>
                {match.calibrated
                  ? "Court calibrated: analysis will track the four players as well as cut clips."
                  : "Clips and highlights work straight away. Calibrate the court once to also track the players."}
              </p>
              <div className="row">
                <button onClick={() => setCalibrating(true)} disabled={busy}>
                  {match.calibrated ? "Recalibrate court" : "Calibrate court"}
                </button>
                <button className="primary" onClick={analyze} disabled={busy}>
                  {match.job ? "Analyse again" : "Analyse"}
                </button>
              </div>
            </>
          )}
          {busy && match.job && (
            <div style={{ marginTop: 16 }}>
              <p className="small muted" style={{ margin: "0 0 6px" }}>
                {match.status === "queued" ? "Waiting for a worker…" : `Analysing… ${Math.round(match.job.progress * 100)}%`}
              </p>
              <ProgressBar value={match.job.progress} />
            </div>
          )}
          {match.status === "failed" && <p className="error small">Analysis failed: {match.error}</p>}
        </section>
      )}

      {results && (
        <MatchViewer
          key={match.job?.id}
          api={api}
          matchId={id}
          durationS={match.duration_s}
          data={results}
          names={names}
          focus={focus}
          seekTo={seekTo}
        />
      )}
      {results?.report && (
        <Report
          api={api}
          matchId={id}
          report={results.report}
          names={names}
          onNames={saveNames}
          focus={focus}
          onFocus={setFocus}
          onSeek={(t) => setSeekTo((prev) => ({ t, n: (prev?.n ?? 0) + 1 }))}
        />
      )}
      {results && <Results api={api} matchId={id} data={results} />}

      {error && <p className="error small">{error}</p>}
      <p>
        <button className="link small" onClick={remove} disabled={busy && match.status === "processing"}>
          Delete match
        </button>
      </p>
    </>
  );
}
