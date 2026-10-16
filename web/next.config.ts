import type { NextConfig } from "next";

// A static export: plain HTML/JS that any web host can serve, and that the
// Android (Capacitor) and desktop (Tauri) shells can bundle as-is later.
// All data comes from the API at runtime.
const config: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default config;
