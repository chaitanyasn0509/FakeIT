import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UNCLOUD IT",
  description: "Cloud removal and surface reconstruction for LISS-IV imagery"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
