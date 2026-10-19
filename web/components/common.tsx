"use client";

import type { MatchStatus } from "@/lib/api";

export function ProgressBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

const LABELS: Record<MatchStatus, string> = {
  created: "Waiting for video",
  uploading: "Uploading",
  uploaded: "Ready to analyse",
  queued: "Queued",
  processing: "Analysing",
  done: "Done",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: MatchStatus }) {
  const cls = status === "done" ? "done" : status === "failed" ? "failed" : status === "processing" ? "running" : "";
  return <span className={`badge ${cls}`}>{LABELS[status]}</span>;
}

export function formatBytes(n: number): string {
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function formatTime(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = String(m).padStart(h ? 2 : 1, "0");
  return `${h ? `${h}:` : ""}${mm}:${String(sec).padStart(2, "0")}`;
}
