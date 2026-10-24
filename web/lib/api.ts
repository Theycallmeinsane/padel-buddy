// Typed client for the Padel Buddy API (api/src/padel_api/main.py).

export type Job = {
  id: string;
  kind?: "analyze" | "relabel"; // relabel: who is who decided again after a tag
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
  player_names?: Record<string, string>; // P1, P2, ... -> name
  tags?: Tag[];
  job: Job | null;
  live?: Live | null; // while analysing, or after a run that stopped: how far the results so far reach
};

/** Results published during an analysis cover the video up to `covered_s`. */
export type Live = { covered_s: number; duration_s: number };

/** Someone named the person at (x, y) (source video pixels) at match time `time_s`. */
export type Tag = {
  id: string;
  time_s: number;
  x: number;
  y: number;
  name: string;
  player: string | null; // the box clicked, when the match was already analysed
  applied: boolean | null; // null until the next analysis; false when nobody tracked was there
};
export type TagIn = Omit<Tag, "id" | "applied">;

export type UploadStatus = { received: number; total: number | null; complete: boolean };
export type Keypoint = { name: string; x_m: number; y_m: number };
export type CalibrationResult = { points_used: string[]; reprojection_error_m: number };

export type PlayerStats = {
  player: string; // P1, P2, ... in the order they first appeared
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
  names?: Record<string, string>; // player -> name, from tags
};

export type Clip = {
  index: number;
  file: string;
  start_s: number;
  end_s: number;
  peak_s: number;
  score: number;
  thumbnail: string | null;
  rallies?: number[]; // the points in the clip; empty for an intense-moment clip
};

export type Rally = {
  index: number;
  start_s: number;
  end_s: number;
  hits: number;
  peak_excess_db: number;
  movement_mps: number | null;
  score: number;
  clip?: string | null; // this point on its own, e.g. points/point_007.mp4
};

export type ClipsSummary = {
  hits: number;
  rallies: Rally[];
  clips: Clip[];
  highlights: string | null;
  highlight_segments: [number, number][];
  condensed?: string | null; // every point back to back
  condensed_s?: number | null;
  condensed_segments?: [number, number, number][]; // match start, match end, start in condensed
};

/** What the viewer draws over the video (the engine's overlay.json). */
export type Overlay = {
  version: number;
  size: [number, number]; // source video pixels, which boxes and lines are in
  fps: number;
  court_to_image: number[][];
  lines: { name: string; points: [number, number][] }[];
  // Sample i: t[i], box[4i..4i+3] = x1 y1 x2 y2, xy[2i..2i+1] = court metres
  // Version 2 has no fixed team: a team is whoever shares a half during a point.
  players: Record<string, { t: number[]; box: number[]; xy: number[] }>;
};

export type BallSummary = {
  searched_s: number;
  ball_frames: number;
  seen_fraction: number;
  flights: number;
  bounces: number;
  outputs: Record<string, string>;
};

export type Bounce = { time_s: number; u: number; v: number; court_x: number; court_y: number };

/** The engine's ball_track.json. */
export type BallTrack = {
  version: number;
  size: [number, number];
  fps: number;
  spans: [number, number][]; // where the ball was looked for (the points)
  // Sample i: t[i], uv[2i..2i+1] in source pixels, flight[i]
  t: number[];
  uv: number[];
  flight: number[];
  bounces: Bounce[];
};

export type ShotKind =
  | "serve" | "return" | "volley" | "smash" | "vibora" | "bandeja" | "lob" | "flat" | "groundstroke" | "unknown";

export type Shot = {
  time_s: number;
  rally: number;
  number: number; // 1 for the serve
  player: string | null; // null when only the team could be told
  team: "A" | "B" | null; // the side the hitter played the point on: A the camera's, B the far one
  kind: ShotKind;
  zone: "net" | "mid" | "back" | null;
  court_x: number | null;
  court_y: number | null;
  by: "ball" | "turn";
  speed_mps?: number | null; // court metres to the next bounce per second in the air, when seen
  // in_play: the other team played it back; winner / error: it ended the point; null: not known
  outcome?: "in_play" | "winner" | "error" | null;
};

export type PointEnding = {
  rally: number;
  start_s: number;
  end_s: number;
  shots: number;
  last_player: string | null;
  last_team: "A" | "B" | null;
  ending: "winner" | "error" | "unknown";
  detail: "net" | "out" | null;
  won_by: "A" | "B" | null;
  teams?: Partial<Record<"A" | "B", string[]>>; // who played the point on each side
};

export type ShotsSummary = { shots: Shot[]; points: PointEnding[]; outputs: Record<string, string> };

export type PlayerMoment = {
  time_s: number;
  start_s: number;
  end_s: number;
  rally: number;
  kind: string; // winner, smash, vibora, bandeja, point
  score: number;
};

/** How one kind of shot went for a player. */
export type ShotStat = {
  kind: ShotKind;
  shots: number;
  known: number; // shots whose outcome is known
  in_play: number; // played back by the other team
  winners: number;
  errors: number;
  success: number | null; // (in play + winners) / known, 0..1
  rating: "strong" | "solid" | "weak" | null; // once there are enough known shots
};

/** A vertical (9:16) clip of one point, for phones and sharing. */
export type Short = {
  file: string;
  rally: number;
  start_s: number;
  end_s: number;
  title: string; // "Point 12" or "P3's winner"
  player: string | null; // set for a player's own short
};

export type PlayerReport = {
  player: string;
  points?: number; // points they played
  points_won?: number; // by their side
  partners?: Record<string, number>; // partner -> points together, most first
  shots: number;
  shot_kinds: Partial<Record<ShotKind, number>>;
  winners: number;
  errors: number;
  errors_by_kind: Record<string, number>;
  distance_per_point_m: number;
  top_speed_mps: number;
  zone_share: Record<string, number>;
  side_share: Record<string, number>;
  highlights: PlayerMoment[];
  reel: string | null;
  shot_stats?: ShotStat[]; // most played first
  strengths?: ShotKind[]; // rated strong, best first
  weaknesses?: ShotKind[]; // rated weak, weakest first
  winning_shots?: PlayerMoment[]; // every point they finished with a winner
  winners_reel?: string | null; // those points joined
};

/** A pair who played points together. */
export type TeamReport = {
  team: string; // "P1+P3"
  players: string[];
  points?: number; // played together
  points_won: number;
  winners: number;
  errors: number;
  formation: Record<"both_net" | "split" | "both_back" | "other", number>;
  lob_share: number;
  opponents?: string[];
};

export type Tip = { who: string; topic: string; text: string };

export type MatchReport = {
  duration_s: number;
  points: number;
  points_with_ending: number;
  shots: number;
  avg_shots_per_point: number;
  avg_point_s: number;
  longest_point: { rally: number; start_s: number; duration_s: number; shots: number } | null;
  teams: TeamReport[];
  players: PlayerReport[];
  tips: Tip[];
  coach_summary: string | null;
  shorts?: Short[];
  outputs: Record<string, string>;
};

/** One player in one match (GET /matches/{id}/players/{P1, P2, ...} and /people/{name}). */
export type PlayerInMatch = {
  match_id: string;
  match_title: string;
  match_date: string;
  player: string;
  name: string | null;
  report: PlayerReport | null;
  shots: Shot[];
  reel: string | null;
  winners_reel?: string | null;
  shorts?: Short[]; // this player's own shorts
};

export type Person = { name: string; matches: number };

export type Results = {
  tracking: TrackingSummary | null;
  clips: ClipsSummary | null;
  ball?: BallSummary | null;
  shots?: ShotsSummary | null;
  report?: MatchReport | null;
};

// -- connection settings (kept in this browser only) -------------------------

const SETTINGS_KEY = "padel-buddy.api";
export type ApiSettings = { baseUrl: string; token: string };

// Served by `padel-buddy`, the app and the API share an origin; under
// `npm run dev` (port 3000) the API is on localhost:8000.
function defaultBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined" && window.location.port !== "3000") return window.location.origin;
  return "http://localhost:8000";
}

export function loadSettings(): ApiSettings {
  const fallback = { baseUrl: defaultBaseUrl(), token: "" };
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
  // `version` changes when an output is rewritten under the same name (live results), so the browser fetches it again.
  fileUrl = (id: string, rel: string, version?: string) =>
    this.media(`/matches/${id}/files/${rel}${version ? `?v=${encodeURIComponent(version)}` : ""}`);
  videoUrl = (id: string) => this.media(`/matches/${id}/video`);
  // Tags and live results rewrite these in place, so always check they are current.
  overlay = (id: string, rel: string) => this.request<Overlay>(`/matches/${id}/files/${rel}`, { cache: "no-cache" });
  ballTrack = (id: string, rel: string) =>
    this.request<BallTrack>(`/matches/${id}/files/${rel}`, { cache: "no-cache" });
  namePlayers = (id: string, names: Record<string, string>) =>
    this.request<Match>(`/matches/${id}/players`, { method: "PUT", body: JSON.stringify({ names }) });
  addTag = (id: string, tag: TagIn) =>
    this.request<Match>(`/matches/${id}/tags`, { method: "POST", body: JSON.stringify(tag) });
  deleteTag = (id: string, tagId: string) => this.request<Match>(`/matches/${id}/tags/${tagId}`, { method: "DELETE" });
  player = (id: string, pid: string) => this.request<PlayerInMatch>(`/matches/${id}/players/${pid}`);
  reportUrl = (id: string) => this.media(`/matches/${id}/report.md`);
  people = () => this.request<Person[]>("/people");
  person = (name: string) => this.request<PlayerInMatch[]>(`/people/${encodeURIComponent(name)}`);
}

/** "Maya (P3)" when the player has a name, else "P3". */
export function playerLabel(pid: string, names?: Record<string, string>): string {
  const name = names?.[pid];
  return name ? `${name} (${pid})` : pid;
}

/** Swap player numbers in engine text (tips, titles) for the players' names; "P1+P3" reads "Maya & Sam". */
export function withNames(text: string, names: Record<string, string> = {}): string {
  return text.replace(/\bP\d+\+P\d+\b/g, (pair) => pairLabel(pair, names)).replace(/\bP\d+\b/g, (pid) => names[pid] ?? pid);
}

/** "Maya & Sam" for the pair "P1+P3", numbers where there are no names. */
export function pairLabel(pair: string, names: Record<string, string> = {}): string {
  return pair
    .split("+")
    .map((pid) => names[pid] ?? pid)
    .join(" & ");
}

/** Players' numbers in order: P2 before P10. */
export function byNumber(a: string, b: string): number {
  return Number(a.slice(1)) - Number(b.slice(1));
}
