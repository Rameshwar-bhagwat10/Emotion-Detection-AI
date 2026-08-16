import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Emotion Detection AI - System Status",
  description: "AI-Based Facial Expression Emotion Detection & Analytics System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background font-sans antialiased">{children}</body>
    </html>
  );
}
