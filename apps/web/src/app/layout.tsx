import type { Metadata } from "next";
import { Saira_Condensed, Cormorant_Garamond, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const fontDisplay = Saira_Condensed({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-display",
  display: "swap",
});

const fontSerif = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-serif",
  display: "swap",
});

const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VALENCE · EMOTION AI",
  description: "Ultra-Precision Facial Affect Intelligence & Telemetry Engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`min-h-screen bg-black text-[#cccccc] ${fontDisplay.variable} ${fontSerif.variable} ${fontMono.variable} font-serif antialiased selection:bg-white selection:text-black`}
      >
        {children}
      </body>
    </html>
  );
}
