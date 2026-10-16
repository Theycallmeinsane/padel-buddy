// Resumable upload of a large match video in chunks.
//
// Each chunk carries Content-Range. If the connection drops, we ask the
// server how many bytes it holds and carry on from there, so a 5 GB upload
// over a flaky phone connection does not restart from zero.

import type { Api, UploadStatus } from "./api";

export const CHUNK_BYTES = 8 * 1024 * 1024;

export type UploadOptions = {
  chunkBytes?: number;
  retries?: number;
  onProgress?: (sent: number, total: number) => void;
  signal?: AbortSignal;
  sleep?: (ms: number) => Promise<void>;
};

type Uploader = Pick<Api, "uploadChunk" | "uploadStatus">;

export async function uploadResumable(
  api: Uploader,
  matchId: string,
  file: Blob,
  opts: UploadOptions = {},
): Promise<UploadStatus> {
  const chunk = opts.chunkBytes ?? CHUNK_BYTES;
  const retries = opts.retries ?? 5;
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const total = file.size;

  let status = await api.uploadStatus(matchId);
  let offset = status.received;
  let failures = 0;
  opts.onProgress?.(offset, total);

  while (!status.complete) {
    if (opts.signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");
    const end = Math.min(offset + chunk, total);
    try {
      status = await api.uploadChunk(matchId, file.slice(offset, end), offset, total);
      offset = status.received;
      failures = 0;
      opts.onProgress?.(offset, total);
    } catch (err) {
      failures += 1;
      if (failures > retries) throw err;
      await sleep(Math.min(30_000, 1000 * 2 ** (failures - 1)));
      // Whatever happened, the server's count is the truth.
      try {
        status = await api.uploadStatus(matchId);
        offset = status.received;
      } catch {
        // Still offline; the next loop retries.
      }
    }
  }
  return status;
}

// Remember which match a local file was being uploaded to, so picking the
// same file again after a reload resumes instead of creating a new match.
const PENDING_KEY = "padel-buddy.pending-uploads";

export function fileKey(f: File): string {
  return `${f.name}|${f.size}|${f.lastModified}`;
}

export function rememberPending(f: File, matchId: string): void {
  try {
    const all = JSON.parse(localStorage.getItem(PENDING_KEY) || "{}");
    all[fileKey(f)] = matchId;
    localStorage.setItem(PENDING_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: resume just won't survive a reload
  }
}

export function pendingMatchFor(f: File): string | null {
  try {
    return JSON.parse(localStorage.getItem(PENDING_KEY) || "{}")[fileKey(f)] ?? null;
  } catch {
    return null;
  }
}

export function forgetPending(f: File): void {
  try {
    const all = JSON.parse(localStorage.getItem(PENDING_KEY) || "{}");
    delete all[fileKey(f)];
    localStorage.setItem(PENDING_KEY, JSON.stringify(all));
  } catch {
    // ignore
  }
}
