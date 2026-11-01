"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Api, ApiError, type Keypoint } from "@/lib/api";
import { formatTime } from "./common";

// Court lines in metres (x 0..10 left to right from the camera, y 0..20 from
// the camera's back wall to the far one). Mirrors padel_engine/court.py.
const LINES: [number, number, number, number][] = [
  [0, 0, 10, 0], [0, 20, 10, 20], [0, 0, 0, 20], [10, 0, 10, 20],
  [0, 3.05, 10, 3.05], [0, 16.95, 10, 16.95],
  [5, 2.85, 5, 17.15],
];

const MAX_ERROR_M = 0.5; // average disagreement of the points beyond which tracking goes wrong

type Props = {
  api: Api;
  matchId: string;
  durationS: number | null;
  onSaved: () => void;
  onCancel: () => void;
};

export default function Calibrator({ api, matchId, durationS, onSaved, onCancel }: Props) {
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [t, setT] = useState(Math.min(5, durationS ?? 5));
  const [size, setSize] = useState<[number, number] | null>(null);
  const [points, setPoints] = useState<Record<string, [number, number]>>({});
  const [order, setOrder] = useState<string[]>([]); // names in the order they were clicked
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Saved, but the points disagree: stay open so they can be fixed, or kept on purpose.
  const [warning, setWarning] = useState<string | null>(null);
  const img = useRef<HTMLImageElement>(null);

  useEffect(() => {
    api.keypoints().then(setKeypoints).catch((e: Error) => setError(e.message));
  }, [api]);

  const current = keypoints[step];
  const frameUrl = useMemo(() => api.frameUrl(matchId, t), [api, matchId, t]);
  const count = Object.keys(points).length;

  function click(e: React.MouseEvent<HTMLImageElement>) {
    if (!current || !img.current) return;
    const r = img.current.getBoundingClientRect();
    const x = ((e.clientX - r.left) * img.current.naturalWidth) / r.width;
    const y = ((e.clientY - r.top) * img.current.naturalHeight) / r.height;
    setPoints({ ...points, [current.name]: [Math.round(x * 10) / 10, Math.round(y * 10) / 10] });
    setOrder([...order.filter((n) => n !== current.name), current.name]);
    setStep(step + 1);
  }

  function undo() {
    const last = order[order.length - 1];
    if (!last) return;
    const { [last]: _, ...rest } = points;
    setPoints(rest);
    setOrder(order.slice(0, -1));
    setStep(keypoints.findIndex((k) => k.name === last));
  }

  async function save() {
    if (!size) return;
    setSaving(true);
    setError(null);
    setWarning(null);
    try {
      const res = await api.saveCalibration(matchId, size, points);
      if (res.reprojection_error_m > MAX_ERROR_M) {
        setWarning(
          `Saved, but the points disagree by ${(res.reprojection_error_m * 100).toFixed(0)} cm on average, ` +
            "so players will be tracked in the wrong places. Check each point is on the right line, " +
            "then save again.",
        );
        return;
      }
      onSaved();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <p className="small muted" style={{ marginTop: 0 }}>
        Click each highlighted point on the frame where the court lines meet the floor. Skip any point you
        can&apos;t see. Four points is the minimum; six or more spread over the court work best.
      </p>
      <div className="calib">
        <div>
          <div className="frame">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={img}
              src={frameUrl}
              alt="Video frame to calibrate on"
              onClick={click}
              onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
            />
            {size &&
              Object.entries(points).map(([name, [x, y]]) => (
                <div key={name} className="pt" style={{ left: `${(x / size[0]) * 100}%`, top: `${(y / size[1]) * 100}%` }}>
                  <span>{name.replaceAll("_", " ")}</span>
                </div>
              ))}
          </div>
          {durationS ? (
            <div className="row small" style={{ marginTop: 8 }}>
              <span className="muted">Frame at {formatTime(t)}</span>
              <input
                className="grow"
                type="range"
                min={0}
                max={Math.max(0, Math.floor(durationS - 1))}
                value={t}
                onChange={(e) => setT(Number(e.target.value))}
                aria-label="Choose a frame without players blocking the lines"
              />
            </div>
          ) : null}
        </div>
        <div>
          <svg className="court-diagram" viewBox="-1.5 -1.5 13 24" role="img" aria-label="Court diagram">
            <rect x={0} y={0} width={10} height={20} fill="var(--surface-2)" />
            {LINES.map(([x1, y1, x2, y2], i) => (
              <line key={i} x1={x1} y1={20 - y1} x2={x2} y2={20 - y2} stroke="var(--muted)" strokeWidth={0.12} />
            ))}
            <line x1={-0.4} y1={10} x2={10.4} y2={10} stroke="var(--text)" strokeWidth={0.2} />
            {keypoints.map((k) => {
              const done = k.name in points;
              const on = current?.name === k.name;
              return (
                <circle
                  key={k.name}
                  cx={k.x_m}
                  cy={20 - k.y_m}
                  r={on ? 0.7 : 0.35}
                  fill={done ? "var(--accent)" : on ? "none" : "var(--muted)"}
                  stroke={on ? "#ff3b30" : "none"}
                  strokeWidth={0.25}
                />
              );
            })}
            <text x={5} y={22} textAnchor="middle" fontSize={0.9} fill="var(--muted)">camera side</text>
          </svg>
          <p style={{ margin: "8px 0" }}>
            {current ? (
              <>
                Click: <strong>{current.name.replaceAll("_", " ")}</strong>
              </>
            ) : (
              <strong>All points done</strong>
            )}
          </p>
          <div className="row small">
            <button onClick={() => setStep(step + 1)} disabled={!current}>Skip</button>
            <button onClick={undo} disabled={!order.length}>Undo</button>
          </div>
          <p className="small muted">{count} point{count === 1 ? "" : "s"} placed</p>
          <div className="row">
            <button className="primary" onClick={save} disabled={count < 4 || saving || !size}>
              {saving ? "Saving…" : "Save"}
            </button>
            {/* After a save with a warning the points are kept, so closing must show them. */}
            <button onClick={warning ? onSaved : onCancel}>Cancel</button>
          </div>
        </div>
      </div>
      {warning && (
        <div className="row small" role="alert">
          <p className="error grow" style={{ margin: 0 }}>{warning}</p>
          <button onClick={onSaved}>Keep these points</button>
        </div>
      )}
      {error && <p className="error small">{error}</p>}
    </div>
  );
}
