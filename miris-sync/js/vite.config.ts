/**
 * Custom Vite config for the Miris FiftyOne plugin.
 *
 * Builds the plugin as a UMD bundle that FiftyOne loads at runtime.
 * Rendering of `MirisStream` fo3d nodes is provided by FiftyOne core; this
 * plugin only ships the sync operator that ingests Miris assets into the
 * current dataset.
 *
 * `@miris-inc/three` is the only Miris SDK bundled here.
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FiftyOne exposes these as globals in the App runtime
const FIFTYONE_GLOBALS: Record<string, string> = {
  react: "React",
  "react-dom": "ReactDOM",
  "@fiftyone/operators": "__foo__",
  three: "__three__",
};

export default defineConfig({
  plugins: [
    react({
      // Use classic JSX transform (React.createElement) instead of the
      // automatic runtime; FiftyOne exposes React but not react/jsx-runtime.
      jsxRuntime: "classic",
    }),
  ],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
    "process.env": JSON.stringify({}),
  },
  build: {
    lib: {
      entry: "src/index.tsx",
      formats: ["umd"],
      name: "MirisSync",
      fileName: () => "index.umd.js",
    },
    outDir: "dist",
    sourcemap: true,
    // Disable minification to prevent UMD factory parameter shadowing
    // (the minifier reuses parameter names that shadow externalized imports).
    minify: false,
    rollupOptions: {
      external: Object.keys(FIFTYONE_GLOBALS),
      output: {
        globals: FIFTYONE_GLOBALS,
      },
    },
  },
});
