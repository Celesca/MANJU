const flowbite = require("flowbite-react/tailwind");

/** @type {import('tailwindcss').Config} */
export default {
  // prefix: "tw-",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}", flowbite.content()],
  theme: {
    extend: {
      colors: {
        green : '#2a9d8f',
        yellow : '#e9c46a',
        orange : '#f4a261',
        red : '#e76f51'
      },
    },
  },
  variants: {},
  plugins: [flowbite.plugin(), require("daisyui")],
  daisyui: {
    themes: ["light"],
  },
};