// Typed client for the Padel Buddy API (api/src/padel_api/main.py).

export type Job = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type MatchStatus =
  | "created" | "uploading" | "uploaded" | "queued" | "processing" | "done" | "failed";

export type Match = {
  id: string;
  title: string;
  created_at: string;
  status: MatchStatus;
  video_size: number | null;
  video_received: number;
  duration_s: number | null;
  calibrated: boolean;
  error: string | null;
  job: Job | null;
};

export type UploadStatus = { received: number; total: number | null; complete: boolean };
export type Keypoint = { name: string; x_m: number; y_m: number };
export type CalibrationResult = { points_used: string[]; reprojection_error_m: number };

export type PlayerStats = {
  player: string;
  team: string;
  seen_fraction: number;
  distance_m: number;
  avg_speed_mps: number;
  top_speed_mps: number;
  zone_share: Record<string, number>;
  side_share: Record<string, number>;
  heatmap: string | null;
};

export type TrackingSummary = {
  players: PlayerStats[];
  analysed_fps: number;
  outputs: Record<string, string>;
};

export type Clip = {
  index: number;
  file: string;
  start_s: number;
  end_s: number;
  peak_s: number;
  score: number;
  thumbnail: string | null;
};

export type Rally = {
  index: number;
  start_s: number;
  end_s: number;
  hits: number;
  peak_excess_db: number;
  movement_mps: number | null;
  score: number;
};

export type ClipsSummary = {
  hits: number;
  rallies: Rally[];
  clips: Clip[];
  highlights: string | null;
  highlight_segments: [number, number][];
};

export type Results = { tracking: TrackingSummary | null; clips: ClipsSummary | null };

// -- connection settings (kept in this browser only) -------------------------

const SETTINGS_KEY = "padel-buddy.api";
export type ApiSettings = { baseUrl: string; token: string };

export function loadSettings(): ApiSettings {
  const fallback = { baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000", token: "" };
  if (typeof window === "undefined") return fallback;
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return fallback;
  }
}

export function saveSettings(s: ApiSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    // Private mode or storage blocked: settings last for this page only.
  }
}

// -- requests -----------------------------------------------------------------

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export class Api {
  // fetch must be called unbound from window, hence the wrapper.
  constructor(
    private s: ApiSettings = loadSettings(),
    private fetchImpl: typeof fetch = (input, init) => fetch(input, init),
  ) {}

  url(path: string): string {
    return this.s.baseUrl.replace(/\/$/, "") + path;
  }

  /** URL for <video>/<img> tags, which cannot send headers: token goes in the query. */
  media(path: string): string {
    const u = this.url(path);
    return this.s.token ? `${u}${u.includes("?") ? "&" : "?"}token=${encodeURIComponent(this.s.token)}` : u;
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (this.s.token) headers.set("Authorization", `Bearer ${this.s.token}`);
    if (init.body && typeof init.body === "string") headers.set("Content-Type", "application/json");
    const res = await this.fetchImpl(this.url(path), { ...init, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        // not JSON
      }
      throw new ApiError(res.status, detail);
    }
    if (res.status === 204) return undefined as T;
    const type = res.headers.get("content-type") || "";
    return (type.includes("application/json") ? res.json() : res.blob()) as Promise<T>;
  }

  health = () => this.request<{ ok: boolean; version: string }>("/health");
  listMatches = () => this.request<Match[]>("/matches");
  getMatch = (id: string) => this.request<Match>(`/matches/${id}`);
  createMatch = (title: string, filename: string, size: number) =>
    this.request<Match>("/matches", { method: "POST", body: JSON.stringify({ title, filename, size }) });
  deleteMatch = (id: string) => this.request<void>(`/matches/${id}`, { method: "DELETE" });
  uploadStatus = (id: string) => this.request<UploadStatus>(`/matches/${id}/video/status`);
  uploadChunk = (id: string, chunk: Blob, start: number, total: number) =>
    this.request<UploadStatus>(`/matches/${id}/video`, {
      method: "PUT",
      body: chunk,
      headers: { "Content-Range": `bytes ${start}-${start + chunk.size - 1}/${total}` },
    });
  keypoints = () => this.request<Keypoint[]>("/court/keypoints");
  saveCalibration = (id: string, imageSize: [number, number], points: Record<string, [number, number]>) =>
    this.request<CalibrationResult>(`/matches/${id}/calibration`, {
      method: "PUT",
      body: JSON.stringify({ image_size: imageSize, points }),
    });
  clearCalibration = (id: string) => this.request<void>(`/matches/${id}/calibration`, { method: "DELETE" });
  analyze = (id: string, config?: object) =>
    this.request<Job>(`/matches/${id}/analyze`, { method: "POST", body: JSON.stringify({ config: config ?? null }) });
  results = (id: string) => this.request<Results>(`/matches/${id}/results`);
  frameUrl = (id: string, t: number) => this.media(`/matches/${id}/frame?t=${t}`);
  fileUrl = (id: string, rel: string) => this.media(`/matches/${id}/files/${rel}`);
}
