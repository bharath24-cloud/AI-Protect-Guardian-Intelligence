import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // AI-Protect Guardian Intelligence Palette
        graphite: {
          950: "#0c0e12",
          DEFAULT: "#101217", // Deep graphite background
          elevated: "#181C24", // Elevated panel
          secondary: "#202631", // Secondary panel
          border: "#262C38",
          subtle: "#1F242F",
        },
        ivory: {
          DEFAULT: "#F4F1E9", // Warm ivory text
          muted: "#D8D4C8",
        },
        teal: {
          electric: "#5CE1D2", // Electric teal (active analysis)
          DEFAULT: "#5CE1D2",
          muted: "#38A89C",
        },
        coral: {
          soft: "#FF8C79", // Soft coral (meaningful warnings)
          DEFAULT: "#FF8C79",
        },
        violet: {
          muted: "#A9A5FF", // Muted violet (secondary AI accents)
          DEFAULT: "#A9A5FF",
        },
        secondary: {
          text: "#9BA5B4", // Secondary text
        },
        // Backward-compatible semantic aliases
        background: "#101217",
        surface: "#181C24",
        "surface-border": "#262C38",
        primary: {
          50: "#eef8ff",
          100: "#d8eeff",
          400: "#5CE1D2",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
        },
        shield: {
          cyan: "#5CE1D2",
          emerald: "#5CE1D2",
          amber: "#FF8C79",
          rose: "#FF8C79",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "ui-monospace",
          "SFMono-Regular",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
export default config;
