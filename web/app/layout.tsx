import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "生成式互动影像",
  description: "你说一个世界，它拍给你看。",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // The player is full-bleed video; letting the page rubber-band over it on iOS
  // shows the page background through the letterbox.
  maximumScale: 1,
  themeColor: "#07070a",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
