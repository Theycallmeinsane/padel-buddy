import { describe, expect, it } from "vitest";
import type { UploadStatus } from "../lib/api";
import { uploadResumable } from "../lib/upload";

// A fake server that stores bytes and enforces the API's "no gaps" rule.
function fakeServer(failOn: Set<number> = new Set()) {
  let received = 0;
  let total: number | null = null;
  let calls = 0;
  const api = {
    async uploadStatus(): Promise<UploadStatus> {
      return { received, total, complete: total !== null && received >= total };
    },
    async uploadChunk(_id: string, chunk: Blob, start: number, size: number): Promise<UploadStatus> {
      calls += 1;
      if (failOn.has(calls)) {
        // Simulate a drop after the server stored half the chunk.
        received = Math.max(received, start + Math.floor(chunk.size / 2));
        throw new Error("network down");
      }
      if (start > received) throw new Error("gap");
      total = size;
      received = Math.max(received, start + chunk.size);
      return { received, total, complete: received >= size };
    },
  };
  return { api, get received() { return received; } };
}

describe("uploadResumable", () => {
  it("uploads in chunks and reports progress", async () => {
    const server = fakeServer();
    const seen: number[] = [];
    const file = new Blob([new Uint8Array(1000)]);
    const out = await uploadResumable(server.api, "m", file, { chunkBytes: 300, onProgress: (s) => seen.push(s) });
    expect(out.complete).toBe(true);
    expect(seen).toEqual([0, 300, 600, 900, 1000]);
  });

  it("resumes from the server's count after a dropped chunk", async () => {
    const server = fakeServer(new Set([2]));
    const file = new Blob([new Uint8Array(1000)]);
    const out = await uploadResumable(server.api, "m", file, { chunkBytes: 300, sleep: async () => {} });
    expect(out.complete).toBe(true);
    expect(server.received).toBe(1000);
  });

  it("gives up after too many failures", async () => {
    const server = fakeServer(new Set([1, 2, 3]));
    const file = new Blob([new Uint8Array(1000)]);
    await expect(
      uploadResumable(server.api, "m", file, { chunkBytes: 300, retries: 2, sleep: async () => {} }),
    ).rejects.toThrow("network down");
  });
});

describe("Api", () => {
  it("sends the token and parses errors", async () => {
    const { Api, ApiError } = await import("../lib/api");
    const calls: [string, RequestInit][] = [];
    const fake = (async (url: string, init: RequestInit) => {
      calls.push([url, init]);
      return new Response(JSON.stringify({ detail: "Match not found" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const api = new Api({ baseUrl: "http://x:8000/", token: "t0k" }, fake);
    await expect(api.getMatch("abc")).rejects.toEqual(new ApiError(404, "Match not found"));
    expect(calls[0][0]).toBe("http://x:8000/matches/abc");
    expect(new Headers(calls[0][1].headers).get("Authorization")).toBe("Bearer t0k");
    expect(api.fileUrl("abc", "clips/clip_001.mp4")).toBe("http://x:8000/matches/abc/files/clips/clip_001.mp4?token=t0k");
  });
});
