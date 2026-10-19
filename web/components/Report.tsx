"use client";

import { useState } from "react";
import {
  playerLabel,
  type Api,
  type MatchReport,
  type PlayerMoment,
  type PlayerReport,
  type ShotKind,
} from "@/lib/api";
import { formatTime } from "./common";
import { PLAYER_COLORS } from "./MatchViewer";

const PLAYERS = ["P1", "P2", "P3", "P4"];
const KIND_LABELS: Record<ShotKind, string> = {
  serve: "Serve",
  return: "Return",
  volley: "Volley",
  smash: "Smash",
  bandeja: "Bandeja",
  lob: "Lob",
  groundstroke: "Groundstroke",
  unknown: "Not sure",
};
const MOMENT_LABELS: Record<string, string> = {
  winner: "Winner",
  smash: "Smash",
  bandeja: "Bandeja",
  point: "Long point won",
};

type Props = {
  api: Api;
  matchId: string;
  report: MatchReport;
  names: Record<string, string>;
  onNames: (names: Record<string, string>) => void;
  focus: string | null;
  onFocus: (pid: string | null) => void;
  onSeek: (t: number) => void;
};

export default function Report({ api, matchId, report, names, onNames, focus, onFocus, onSeek }: Props) {
  const label = (pid: string) => playerLabel(pid, names);
  const known = report.points_with_ending;
  return (
    <>
      <section className="card">
        <div className="row" style={{ marginBottom: 4 }}>
          <h2 className="grow" style={{ margin: 0 }}>
            Match report
          </h2>
          <a className="small" href={api.reportUrl(matchId)} download="match-report.md">
            Download
          </a>
        </div>
        <p className="muted small" style={{ marginTop: 0 }}>
          {report.points} points · {report.shots} shots · a point lasted {Math.round(report.avg_point_s)} s and{" "}
          {Math.round(report.avg_shots_per_point)} shots on average
          {report.longest_point
            ? ` · longest ${Math.round(report.longest_point.duration_s)} s at ${formatTime(report.longest_point.start_s)}`
            : ""}
        </p>

        <PlayerNames names={names} onSave={onNames} />

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Team</th>
                <th>Points won</th>
                <th>Winners</th>
                <th>Errors</th>
                <th>Both at net</th>
                <th>Split</th>
                <th>Both back</th>
                <th>In between</th>
              </tr>
            </thead>
            <tbody>
              {report.teams.map((t) => (
                <tr key={t.team}>
                  <td className={t.team === "A" ? "team-a" : "team-b"}>{t.players.map(label).join(" & ")}</td>
                  <td>{t.points_won}</td>
                  <td>{t.winners}</td>
                  <td>{t.errors}</td>
                  <td>{pct(t.formation.both_net)}</td>
                  <td>{pct(t.formation.split)}</td>
                  <td>{pct(t.formation.both_back)}</td>
                  <td>{pct(t.formation.other)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {known < report.points && (
          <p className="muted small">
            How a point ended (winner or error) is known for {known} of {report.points} points, where the ball was seen
            well enough.
          </p>
        )}

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>Shots</th>
                <th>Winners</th>
                <th>Errors</th>
                <th>At the net</th>
                <th>Per point</th>
                <th>Top speed</th>
              </tr>
            </thead>
            <tbody>
              {report.players.map((p) => (
                <tr key={p.player}>
                  <td>
                    <button className="link" onClick={() => onFocus(p.player)} style={{ color: PLAYER_COLORS[p.player] }}>
                      {label(p.player)}
                    </button>
                  </td>
                  <td>{p.shots}</td>
                  <td>{p.winners}</td>
                  <td>{p.errors}</td>
                  <td>{pct(p.zone_share.net)}</td>
                  <td>{Math.round(p.distance_per_point_m)} m</td>
                  <td>{(p.top_speed_mps * 3.6).toFixed(1)} km/h</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">Movement counts only time during points. Click a player to see their highlights.</p>

        {report.tips.length > 0 && (
          <>
            <h3>What to work on</h3>
            <ul className="tips">
              {report.tips.map((t, i) => (
                <li key={i}>{withNames(t.text, names)}</li>
              ))}
            </ul>
          </>
        )}
        {report.coach_summary && (
          <>
            <h3>Coach&apos;s summary</h3>
            <p className="coach">{withNames(report.coach_summary, names)}</p>
          </>
        )}
      </section>

      <PlayerFocus
        api={api}
        matchId={matchId}
        report={report}
        names={names}
        focus={focus}
        onFocus={onFocus}
        onSeek={onSeek}
      />
    </>
  );
}

function PlayerNames({ names, onSave }: { names: Record<string, string>; onSave: (n: Record<string, string>) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(names);
  if (!editing) {
    return (
      <p className="small" style={{ marginTop: 0 }}>
        <button
          className="link"
          onClick={() => {
            setDraft(names);
            setEditing(true);
          }}
        >
          {Object.keys(names).length ? "Rename players" : "Name the players"}
        </button>{" "}
        <span className="muted">so their stats and highlights follow them across matches.</span>
      </p>
    );
  }
  return (
    <form
      className="names"
      onSubmit={(e) => {
        e.preventDefault();
        onSave(Object.fromEntries(PLAYERS.map((pid) => [pid, draft[pid] ?? ""])));
        setEditing(false);
      }}
    >
      {PLAYERS.map((pid) => (
        <div key={pid}>
          <label htmlFor={`name-${pid}`} style={{ color: PLAYER_COLORS[pid] }}>
            {pid} · team {pid === "P1" || pid === "P2" ? "A" : "B"}
          </label>
          <input
            id={`name-${pid}`}
            value={draft[pid] ?? ""}
            maxLength={60}
            placeholder="Name"
            onChange={(e) => setDraft({ ...draft, [pid]: e.target.value })}
          />
        </div>
      ))}
      <div className="row">
        <button className="primary" type="submit">
          Save names
        </button>
        <button type="button" onClick={() => setEditing(false)}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function PlayerFocus({
  api,
  matchId,
  report,
  names,
  focus,
  onFocus,
  onSeek,
}: {
  api: Api;
  matchId: string;
  report: MatchReport;
  names: Record<string, string>;
  focus: string | null;
  onFocus: (pid: string | null) => void;
  onSeek: (t: number) => void;
}) {
  const p = report.players.find((x) => x.player === focus) ?? null;
  return (
    <section className="card" id="player">
      <div className="row" style={{ marginBottom: 12 }}>
        <h2 className="grow" style={{ margin: 0 }}>
          Player highlights
        </h2>
        <div className="segmented" role="group" aria-label="Player">
          {report.players.map((x) => (
            <button
              key={x.player}
              className={x.player === focus ? "on" : ""}
              onClick={() => onFocus(x.player === focus ? null : x.player)}
            >
              {names[x.player] ?? x.player}
            </button>
          ))}
        </div>
      </div>
      {!p ? (
        <p className="muted" style={{ margin: 0 }}>
          Pick a player to see their own highlight reel, best moments and numbers. The viewer above then follows them.
        </p>
      ) : (
        <PlayerCard api={api} matchId={matchId} p={p} name={playerLabel(p.player, names)} onSeek={onSeek} />
      )}
    </section>
  );
}

export function PlayerCard({
  api,
  matchId,
  p,
  name,
  onSeek,
}: {
  api: Api;
  matchId: string;
  p: PlayerReport;
  name: string;
  onSeek?: (t: number) => void;
}) {
  const kinds = Object.entries(p.shot_kinds) as [ShotKind, number][];
  const most = Math.max(1, ...kinds.map(([, n]) => n));
  return (
    <div className="player-card">
      <div className="stats">
        <Stat label="Shots" value={p.shots} />
        <Stat label="Winners" value={p.winners} />
        <Stat label="Errors" value={p.errors} />
        <Stat label="At the net" value={pct(p.zone_share.net)} />
        <Stat label="Per point" value={`${Math.round(p.distance_per_point_m)} m`} />
        <Stat label="Top speed" value={`${(p.top_speed_mps * 3.6).toFixed(1)} km/h`} />
      </div>
      {p.reel ? (
        <video className="reel" controls preload="metadata" src={api.fileUrl(matchId, p.reel)} />
      ) : (
        <p className="muted small">
          {p.highlights.length ? "No reel was cut for this player." : `No winners or overheads were found for ${name}.`}
        </p>
      )}
      {kinds.length > 0 && (
        <div className="kinds" aria-label={`${name}'s shots by kind`}>
          {kinds.map(([k, n]) => (
            <div key={k} className="kind">
              <span>{KIND_LABELS[k] ?? k}</span>
              <span className="bar">
                <span style={{ width: `${(100 * n) / most}%`, background: PLAYER_COLORS[p.player] }} />
              </span>
              <span className="muted">{n}</span>
            </div>
          ))}
        </div>
      )}
      {Object.keys(p.errors_by_kind).length > 0 && (
        <p className="small muted">
          Errors came from:{" "}
          {Object.entries(p.errors_by_kind)
            .map(([k, n]) => `${n} ${(KIND_LABELS[k as ShotKind] ?? k).toLowerCase()}`)
            .join(", ")}
          .
        </p>
      )}
      {p.highlights.length > 0 && (
        <ul className="moments">
          {p.highlights.map((m) => (
            <Moment key={`${m.rally}-${m.time_s}`} m={m} onSeek={onSeek} />
          ))}
        </ul>
      )}
    </div>
  );
}

function Moment({ m, onSeek }: { m: PlayerMoment; onSeek?: (t: number) => void }) {
  const text = `${MOMENT_LABELS[m.kind] ?? m.kind} · point ${m.rally + 1} · ${formatTime(m.time_s)}`;
  return (
    <li>
      {onSeek ? (
        <button className="link" onClick={() => onSeek(m.start_s)}>
          ▶ {text}
        </button>
      ) : (
        text
      )}
    </li>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span className="muted small">{label}</span>
    </div>
  );
}

function withNames(text: string, names: Record<string, string>): string {
  return text.replace(/\bP[1-4]\b/g, (pid) => names[pid] ?? pid);
}

function pct(x: number | undefined): string {
  return x === undefined ? "–" : `${Math.round(x * 100)}%`;
}
