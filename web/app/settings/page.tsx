"use client";

import { useEffect, useState } from "react";
import { Api, loadSettings, saveSettings, type ApiSettings } from "@/lib/api";

export default function SettingsPage() {
  const [s, setS] = useState<ApiSettings>({ baseUrl: "", token: "" });
  const [check, setCheck] = useState<string | null>(null);

  useEffect(() => setS(loadSettings()), []);

  async function save() {
    saveSettings(s);
    setCheck("Checking…");
    try {
      const h = await new Api(s).health();
      await new Api(s).listMatches(); // also proves the token
      setCheck(`Connected to API ${h.version}.`);
    } catch (e) {
      setCheck(`Saved, but the API didn't answer: ${(e as Error).message}`);
    }
  }

  return (
    <>
      <h1>Settings</h1>
      <section className="card">
        <h2>API connection</h2>
        <p className="muted small" style={{ marginTop: 0 }}>
          Stored in this browser only. On your own computer the default is fine; on a phone use your
          computer&apos;s address on the local network (for example http://192.168.1.20:8000).
        </p>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="url">API address</label>
          <input id="url" type="url" value={s.baseUrl} onChange={(e) => setS({ ...s, baseUrl: e.target.value })} />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="token">API token (only if the API sets PADEL_API_TOKEN)</label>
          <input id="token" type="password" value={s.token} onChange={(e) => setS({ ...s, token: e.target.value })} />
        </div>
        <div className="row">
          <button className="primary" onClick={save}>Save and test</button>
          {check && <span className="small muted">{check}</span>}
        </div>
      </section>
    </>
  );
}
