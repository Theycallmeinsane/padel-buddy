"use client";

import { useState } from "react";
import type { Api, Clip, Results as ResultsData } from "@/lib/api";
import { formatTime } from "./common";

export default function Results({ api, matchId, data }: { api: Api; matchId: string; data: ResultsData }) {
  const clips = data.clips;
  const tracking = data.tracking;
  const points = clips?.clips.some((c) => c.rallies?.length) ?? false;
  return (
    <>
      {clips && (
        <section className="card">
          <h2>Highlights</h2>
          <p className="muted small" style={{ marginTop: 0 }}>
            {clips.hits} shots heard · {clips.rallies.length} points · {plural(clips.clips.length, points ? "best point" : "intense moment")}
          </p>
          {clips.highlights ? (
            <video className="reel" controls preload="metadata" src={api.fileUrl(matchId, clips.highlights)} />
          ) : (
            <p className="muted">No rallies were found in the sound, so there is no reel.</p>
          )}
          {clips.condensed && (
            <details style={{ marginTop: 12 }}>
              <summary>
                Condensed match: every point back to back
                {clips.condensed_s ? ` (${formatTime(clips.condensed_s)})` : ""}
              </summary>
              <video className="reel" controls preload="none" src={api.fileUrl(matchId, clips.condensed)} />
            </details>
          )}
        </section>
      )}

      {clips && clips.clips.length > 0 && (
        <section className="card">
          <h2>{points ? "Best points" : "Clips"}</h2>
          <div className="grid">
            {clips.clips.map((c) => (
              <ClipCard key={c.index} api={api} matchId={matchId} clip={c} />
            ))}
          </div>
        </section>
      )}

      {tracking && (
        <section className="card">
          <h2>Players</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Seen</th>
                  <th>Distance</th>
                  <th>Avg speed</th>
                  <th>Top speed</th>
                  <th>Net</th>
                  <th>Mid</th>
                  <th>Back</th>
                  <th>Left side</th>
                </tr>
              </thead>
              <tbody>
                {tracking.players.map((p) => (
                  <tr key={p.player}>
                    <td className={p.team === "A" ? "team-a" : "team-b"}>
                      {p.player} <span className="muted small">team {p.team}</span>
                    </td>
                    <td>{pct(p.seen_fraction)}</td>
                    <td>{p.distance_m.toFixed(0)} m</td>
                    <td>{(p.avg_speed_mps * 3.6).toFixed(1)} km/h</td>
                    <td>{(p.top_speed_mps * 3.6).toFixed(1)} km/h</td>
                    <td>{pct(p.zone_share.net)}</td>
                    <td>{pct(p.zone_share.mid)}</td>
                    <td>{pct(p.zone_share.back)}</td>
                    <td>{pct(p.side_share.left)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            P1/P2 started on the camera side. Net is within 3.5 m of the net, back is behind the service line.
          </p>
          <div className="heatmaps">
            {tracking.players
              .filter((p) => p.heatmap)
              .map((p) => (
                <figure key={p.player}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={api.fileUrl(matchId, p.heatmap!)} alt={`${p.player} heatmap`} />
                  <figcaption className="small">{p.player}</figcaption>
                </figure>
              ))}
          </div>
          {tracking.outputs.preview && (
            <details style={{ marginTop: 12 }}>
              <summary>Tracking preview video</summary>
              <video className="reel" controls preload="none" src={api.fileUrl(matchId, tracking.outputs.preview)} />
            </details>
          )}
        </section>
      )}
    </>
  );
}

function ClipCard({ api, matchId, clip }: { api: Api; matchId: string; clip: Clip }) {
  const [playing, setPlaying] = useState(false);
  return (
    <div className="clip">
      {playing || !clip.thumbnail ? (
        <video controls autoPlay={playing} preload="metadata" src={api.fileUrl(matchId, clip.file)} />
      ) : (
        <button
          className="link"
          style={{ display: "block", width: "100%" }}
          onClick={() => setPlaying(true)}
          aria-label={`Play clip ${clip.index}`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={api.fileUrl(matchId, clip.thumbnail)} alt="" />
        </button>
      )}
      <div className="meta">
        <span>
          {formatTime(clip.start_s)} – {formatTime(clip.end_s)}
        </span>
        <span className="muted">
          {clip.rallies?.length ? plural(clip.rallies.length, "point") : `peak ${formatTime(clip.peak_s)}`} ·{" "}
          {formatTime(clip.end_s - clip.start_s)}
        </span>
      </div>
    </div>
  );
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function pct(x: number | undefined): string {
  return x === undefined ? "–" : `${Math.round(x * 100)}%`;
}
