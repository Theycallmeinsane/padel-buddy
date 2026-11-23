"use client";

import { useRef, useState } from "react";
import { convex, type Anchor, type Pt } from "@/lib/courtAdjust";

type Drag = { kind: "corner" | "edge"; i: number; start: Pt; from: Pt[] };

/**
 * Four handles over the video, on court points the calibration placed. Dragging
 * a corner, or an edge between two corners, moves the drawn court lines with
 * them. Works in source video pixels, like the overlay: the SVG letterboxes
 * itself the same way the video does.
 */
export default function CourtAdjuster({
  size,
  anchors,
  handles,
  onChange,
}: {
  size: Pt;
  anchors: Anchor[];
  handles: Pt[]; // where each anchor is drawn now
  onChange: (to: Pt[]) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const [selected, setSelected] = useState(0);
  const r = Math.max(size[0], size[1]) / 120; // handle radius, in source pixels

  function toSource(e: React.PointerEvent): Pt | null {
    const m = svg.current?.getScreenCTM();
    if (!m) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return [p.x, p.y];
  }

  function commit(next: Pt[]) {
    if (convex(next)) onChange(next.map(([u, v]) => [Math.round(u * 10) / 10, Math.round(v * 10) / 10]));
  }

  function down(e: React.PointerEvent, kind: Drag["kind"], i: number) {
    const at = toSource(e);
    if (!at) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { kind, i, start: at, from: handles };
    if (kind === "corner") setSelected(i);
  }

  function move(e: React.PointerEvent) {
    const d = drag.current;
    const at = d && toSource(e);
    if (!d || !at) return;
    const dx = at[0] - d.start[0];
    const dy = at[1] - d.start[1];
    const moved = d.kind === "corner" ? [d.i] : [d.i, (d.i + 1) % d.from.length];
    commit(d.from.map(([u, v], k) => (moved.includes(k) ? [u + dx, v + dy] : [u, v])));
  }

  function up() {
    drag.current = null;
  }

  function key(e: React.KeyboardEvent, i: number) {
    const step = e.shiftKey ? 10 : 1;
    const by: Record<string, Pt> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const d = by[e.key];
    if (!d) return;
    e.preventDefault();
    commit(handles.map(([u, v], k) => (k === i ? [u + d[0], v + d[1]] : [u, v])));
  }

  return (
    <svg
      ref={svg}
      className="court-adjust"
      viewBox={`0 0 ${size[0]} ${size[1]}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
    >
      <polygon points={handles.map((p) => p.join(",")).join(" ")} className="outline" />
      {handles.map((p, i) => {
        const q = handles[(i + 1) % handles.length];
        return (
          <line
            key={`e${i}`}
            x1={p[0]}
            y1={p[1]}
            x2={q[0]}
            y2={q[1]}
            className="edge"
            style={{ strokeWidth: r * 1.6 }}
            onPointerDown={(e) => down(e, "edge", i)}
          >
            <title>Drag to move this edge</title>
          </line>
        );
      })}
      {handles.map(([u, v], i) => (
        <g key={anchors[i].name}>
          <circle
            cx={u}
            cy={v}
            r={r}
            className={`corner${selected === i ? " on" : ""}`}
            style={{ strokeWidth: r / 3 }}
            tabIndex={0}
            role="slider"
            aria-label={`${anchors[i].name}: drag, or use the arrow keys`}
            aria-valuetext={`${Math.round(u)}, ${Math.round(v)}`}
            onFocus={() => setSelected(i)}
            onKeyDown={(e) => key(e, i)}
            onPointerDown={(e) => down(e, "corner", i)}
          />
          {/* Labels point inwards, so the ones on the right stay in the picture. */}
          <text
            x={u > size[0] / 2 ? u - r * 1.4 : u + r * 1.4}
            y={v - r * 1.2}
            textAnchor={u > size[0] / 2 ? "end" : "start"}
            style={{ fontSize: r * 1.5 }}
          >
            {anchors[i].name}
          </text>
        </g>
      ))}
    </svg>
  );
}
