import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/features/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["var(--font-display)", "Saira Condensed", "-apple-system", "sans-serif"],
        serif: ["var(--font-serif)", "Cormorant Garamond", "Garamond", "serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      letterSpacing: {
        wordmark: "6px",
        "display-xl": "4px",
        "display-lg": "3px",
        "display-md": "2px",
        button: "2.5px",
        caption: "2px",
        nav: "2px",
      },
      colors: {
        // Luxury-Automotive Precision Dark Theme Palette
        canvas: "#000000",
        "surface-soft": "#0d0d0d",
        "surface-card": "#141414",
        "surface-elevated": "#1f1f1f",
        hairline: "#262626",
        "hairline-strong": "#3a3a3a",
        link: "#c3d9f3",
        "body-strong": "#e6e6e6",
        "muted-soft": "#666666",
        background: "#000000",
        foreground: "#ffffff",
        card: {
          DEFAULT: "#141414",
          foreground: "#ffffff",
        },
        popover: {
          DEFAULT: "#1f1f1f",
          foreground: "#ffffff",
        },
        primary: {
          DEFAULT: "#ffffff",
          foreground: "#000000",
        },
        secondary: {
          DEFAULT: "#0d0d0d",
          foreground: "#cccccc",
        },
        muted: {
          DEFAULT: "#141414",
          foreground: "#999999",
        },
        accent: {
          DEFAULT: "#1f1f1f",
          foreground: "#ffffff",
        },
        destructive: {
          DEFAULT: "#7f1d1d",
          foreground: "#ffffff",
        },
        border: "#262626",
        input: "#3a3a3a",
        ring: "#ffffff",
      },
      borderRadius: {
        none: "0px",
        pill: "9999px",
        full: "9999px",
      },
    },
  },
  plugins: [],
} satisfies Config;
