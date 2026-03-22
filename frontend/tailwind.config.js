/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0e0e10",
        surface: "#19191d",
        "surface-variant": "#25252b",
        primary: "#d0bcff",
        "primary-deep": "#6e3bd7",
        "text-primary": "#e7e4ec",
        "text-muted": "#acaab1",
        outline: "#47474d",
        sigma: {
          50:  "#f6f4fb",
          100: "#eae5f3",
          200: "#d8cee9",
          300: "#c3b4dd",
          400: "#a99ac9",
          500: "#8b7cae",
          600: "#6e3bd7",
          700: "#552c9f",
          800: "#3d2e58",
          900: "#25252b",
          950: "#0e0e10",
        },
        neon: {
          cyan:   "#d0bcff",
          violet: "#6e3bd7",
          green:  "#8ddcc2",
          amber:  "#ecb96f",
        },
      },
      fontFamily: {
        display: ["'Inter'", "sans-serif"],
        mono:    ["'JetBrains Mono'", "monospace"],
        body:    ["'Inter'", "sans-serif"],
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(rgba(59,110,246,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(59,110,246,.06) 1px, transparent 1px)",
        "hero-gradient": "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,110,246,0.25), transparent)",
        "card-gradient": "linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%)",
      },
      backgroundSize: {
        "grid-size": "40px 40px",
      },
      animation: {
        "float": "float 6s ease-in-out infinite",
        "pulse-slow": "pulse 4s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
        "spin-slow": "spin 8s linear infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};
