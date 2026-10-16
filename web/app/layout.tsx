import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Padel Buddy",
  description: "Upload a padel match; get clips, highlights and player stats.",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7f4" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1412" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="top">
            <Link href="/" className="brand">🎾 Padel Buddy</Link>
            <nav>
              <Link href="/">Matches</Link>
              <Link href="/settings/">Settings</Link>
            </nav>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
