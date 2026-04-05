/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Sora", "system-ui", "sans-serif"],
      },
      colors: {
        surface: {
          DEFAULT: "#0f1419",
          card: "#1a2332",
          border: "#2d3a4d",
        },
      },
    },
  },
  plugins: [],
};
