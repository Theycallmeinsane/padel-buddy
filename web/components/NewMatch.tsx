"use client";

import { useRef, useState } from "react";
import { Api, ApiError } from "@/lib/api";
import { forgetPending, pendingMatchFor, rememberPending, uploadResumable } from "@/lib/upload";
import { ProgressBar, formatBytes } from "./common";

export default function NewMatch({ onUploaded }: { onUploaded: (matchId: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [sent, setSent] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  function pick(f: File | null) {
    setFile(f);
    setError(null);
    setSent(0);
    if (f && !title) setTitle(f.name.replace(/\.[^.]+$/, ""));
  }

  async function start() {
    if (!file) return;
    const api = new Api();
    setBusy(true);
    setError(null);
    abort.current = new AbortController();
    try {
      // Same file picked again after a reload or a drop: continue that upload.
      let matchId = pendingMatchFor(file);
      if (matchId) {
        try {
          const m = await api.getMatch(matchId);
          if (m.status !== "created" && m.status !== "uploading") matchId = null;
        } catch {
          matchId = null;
        }
      }
      if (!matchId) {
        matchId = (await api.createMatch(title || file.name, file.name, file.size)).id;
        rememberPending(file, matchId);
      }
      await uploadResumable(api, matchId, file, {
        onProgress: (s) => setSent(s),
        signal: abort.current.signal,
      });
      forgetPending(file);
      onUploaded(matchId);
    } catch (err) {
      if ((err as Error).name === "AbortError") setError("Upload paused. Pick the same file again to resume.");
      else if (err instanceof ApiError) setError(err.message);
      else setError(`Upload interrupted (${(err as Error).message}). Pick the same file again to resume.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <h2>New match</h2>
      <div className="row" style={{ marginBottom: 12 }}>
        <div className="grow">
          <label htmlFor="video">Match video</label>
          <input id="video" type="file" accept="video/*" disabled={busy} onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        </div>
      </div>
      {file && (
        <>
          <div style={{ marginBottom: 12 }}>
            <label htmlFor="title">Title</label>
            <input id="title" type="text" value={title} disabled={busy} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div className="row">
            {!busy ? (
              <button className="primary" onClick={start}>Upload</button>
            ) : (
              <button onClick={() => abort.current?.abort()}>Pause</button>
            )}
            <span className="muted small">
              {formatBytes(sent)} of {formatBytes(file.size)}
            </span>
          </div>
          {(busy || sent > 0) && (
            <div style={{ marginTop: 12 }}>
              <ProgressBar value={file.size ? sent / file.size : 0} />
            </div>
          )}
        </>
      )}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
