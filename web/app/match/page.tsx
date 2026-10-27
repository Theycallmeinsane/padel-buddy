"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Calibrator from "@/components/Calibrator";
import MatchViewer from "@/components/MatchViewer";
import Report from "@/components/Report";
import Results from "@/components/Results";
import { PlayerTagger } from "@/components/Tags";
import { ProgressBar, StatusBadge, formatBytes, formatTime } from "@/components/common";
import { Api, ApiError, type Match, type Results as ResultsData, type TagIn } from "@/lib/api";
import { liveNames, suggestions } from "@/lib/tags";

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
  const [naming, setNaming] = useState(false);
  const [people, setPeople] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [seekTo, setSeekTo] = useState<{ t: number; n: number } | null>(null);
  const fetchedTo = useRef<number | null>(null); // how far the live results we hold reach
  const live = match?.live ?? null;
  const tagged = results?.tracking?.names;
  // While analysing, tags given before it started already name players in the results so far.
  const names = useMemo(
    () => (live && tagged ? liveNames(match?.player_names ?? {}, tagged) : match?.player_names ?? {}),
    [live, tagged, match?.player_names],
  );
  const tags = useMemo(() => match?.tags ?? [], [match?.tags]);
  const known = useMemo(() => suggestions(names, people), [names, people]);
  const seek = useCallback((t: number) => setSeekTo((prev) => ({ t, n: (prev?.n ?? 0) + 1 })), []);

  const refresh = useCallback(async () => {
    try {
      const m = await api.getMatch(id);
      setMatch(m);
      // Results so far are fetched again each time they reach further into the match.
      const reach = m.live?.covered_s ?? null;
      if (m.status === "done") {
        fetchedTo.current = null;
        setResults(await api.results(id));
      } else if (reach !== null && reach !== fetchedTo.current) {
        fetchedTo.current = reach;
        setResults(await api.results(id));
      }
    } catch (e) {
      setError(e instanceof ApiError && e.status === 404 ? "This match doesn't exist." : (e as Error).message);
    }
  }, [api, id]);

  useEffect(() => {
    if (id) refresh();
  }, [id, refresh]);

  // Everyone named in earlier matches, offered when naming players here.
  useEffect(() => {
    api.people().then((list) => setPeople(list.map((p) => p.name)), () => {});
  }, [api]);

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
      fetchedTo.current = null;
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

  async function addTag(tag: TagIn): Promise<string | null> {
    try {
      setMatch(await api.addTag(id, tag));
      return null;
    } catch (e) {
      return (e as Error).message;
    }
  }

  async function deleteTag(tagId: string) {
    try {
      setMatch(await api.deleteTag(id, tagId));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function stopAnalysis() {
    if (!confirm("Stop the analysis? What it has found so far stays; analyse again to do the whole match.")) return;
    setError(null);
    try {
      setMatch(await api.stop(id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove() {
    if (!confirm("Delete this match and all its clips?")) return;
    try {
      // While it is being analysed, the first delete stops the analysis; delete again once it has.
      for (let tries = 0; ; tries++) {
        try {
          await api.deleteMatch(id);
          break;
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 409) || tries >= 30) throw e;
          await new Promise((done) => setTimeout(done, 2000));
        }
      }
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
  const relabelling = busy && match.job?.kind === "relabel";
  const stopping = match.job?.status === "stopping";
  const step = match.status === "done" || results ? 3 : match.calibrated || match.job ? 2 : 1;
  const version = `${match.job?.id}:${match.job?.finished_at}:${live?.covered_s ?? ""}`;

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
          {naming ? (
            <PlayerTagger
              api={api}
              matchId={id}
              durationS={match.duration_s}
              tags={tags}
              suggestions={known}
              onTag={addTag}
              onDelete={deleteTag}
              onDone={() => setNaming(false)}
            />
          ) : calibrating ? (
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
                {match.calibrated && (
                  <button onClick={() => setNaming(true)}>
                    {tags.length ? `Named players (${new Set(tags.map((t) => t.name.toLowerCase())).size})` : "Name players"}
                  </button>
                )}
                <button className="primary" onClick={analyze} disabled={busy}>
                  {match.job ? "Analyse again" : "Analyse"}
                </button>
              </div>
            </>
          )}
          {busy && match.job && (
            <div style={{ marginTop: 16 }}>
              <div className="row" style={{ margin: "0 0 6px" }}>
                <p className="small muted grow" style={{ margin: 0 }}>
                  {stopping
                    ? "Stopping…"
                    : relabelling
                      ? `Updating who is who from your tags… ${Math.round(match.job.progress * 100)}%`
                      : match.status === "queued"
                        ? "Waiting for a worker…"
                        : `Analysing… ${Math.round(match.job.progress * 100)}%`}
                </p>
                <button className="small" onClick={stopAnalysis} disabled={stopping}>
                  Stop
                </button>
              </div>
              <ProgressBar value={match.job.progress} />
            </div>
          )}
          {match.status === "stopped" && !live && (
            <p className="small muted">The analysis was stopped. Analyse again when you&apos;re ready.</p>
          )}
          {live && !relabelling && <LiveNote covered={live.covered_s} duration={live.duration_s} running={busy} />}
          {match.status === "failed" && <p className="error small">Analysis failed: {match.error}</p>}
        </section>
      )}

      {results && (
        <MatchViewer
          api={api}
          matchId={id}
          durationS={match.duration_s}
          data={results}
          names={names}
          focus={focus}
          seekTo={seekTo}
          tags={tags}
          suggestions={known}
          onTag={addTag}
          onDeleteTag={deleteTag}
          updating={relabelling}
          live={!!live}
          version={version}
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
          onSeek={seek}
        />
      )}
      {results && (
        <Results
          api={api}
          matchId={id}
          data={results}
          names={names}
          onSeek={seek}
          version={live ? version : undefined}
        />
      )}

      {error && <p className="error small">{error}</p>}
      <p>
        <button className="link small" onClick={remove}>
          {busy ? "Stop and delete match" : "Delete match"}
        </button>
      </p>
    </>
  );
}

/** What the results so far cover, and what is still to come. */
function LiveNote({ covered, duration, running }: { covered: number; duration: number; running: boolean }) {
  if (!running) {
    return (
      <p className="small muted live-note">
        The analysis stopped {covered > 0 ? `at ${formatTime(covered)}` : "early"}. What it found up to there is
        below; analyse again to do the whole match.
      </p>
    );
  }
  return (
    <p className="small muted live-note">
      {covered > 0
        ? `Results so far: the first ${formatTime(covered)} of ${formatTime(duration)}. They update every few minutes.`
        : "The points are in. Player stats, shots and the report appear once the first few minutes are tracked."}{" "}
      The best points, reels and shorts come when the analysis finishes, and player numbers can still change then.
    </p>
  );
}
