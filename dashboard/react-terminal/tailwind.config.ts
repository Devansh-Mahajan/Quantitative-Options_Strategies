import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#0A0A0A",
          panel: "#0F0F0F",
          grid: "#1A1A1A",
          ink: "#E5E7EB",
          muted: "#6B7280",
          cyan: "#00E5FF",
          emerald: "#00FF88",
          crimson: "#FF335F",
          amber: "#F5C542"
        }
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "SFMono-Regular", "Consolas", "monospace"]
      }
    }
  },
  plugins: []
} satisfies Config;
