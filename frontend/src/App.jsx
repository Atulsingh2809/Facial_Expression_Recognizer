import React, { useEffect, useState } from "react";
import { baseURL, checkHealth } from "./api";
import ImageUpload from "./components/ImageUpload";
import WebcamCapture from "./components/WebcamCapture";

/**
 * Root app: mode toggle, health banner, webcam vs upload.
 */
export default function App() {
  const [mode, setMode] = useState("webcam");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealth({ status: "error", model_loaded: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen bg-surface text-slate-100">
      <header className="border-b border-surface-border bg-surface-card/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Facial Expression Recognition
            </h1>
            <p className="mt-1 font-sans text-sm text-slate-400">
              Seven classes · FER2013 · Live or upload
            </p>
          </div>
          <div className="font-sans text-xs text-slate-500">
            <p>
              API:{" "}
              <code className="rounded bg-slate-800 px-1.5 py-0.5 text-cyan-300/90">
                {baseURL}
              </code>
            </p>
            {health && (
              <p className="mt-1">
                <span
                  className={
                    health.status === "ok"
                      ? "text-emerald-400"
                      : "text-amber-400"
                  }
                >
                  {health.status === "ok" ? "●" : "○"}
                </span>{" "}
                {health.status === "ok" ? "API online" : "API unreachable"}{" "}
                {health.model_loaded ? "· model loaded" : "· model not loaded"}
              </p>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-8 flex flex-wrap gap-2 rounded-xl bg-surface-card p-1.5">
          {[
            { id: "webcam", label: "Webcam" },
            { id: "upload", label: "Upload" },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setMode(t.id)}
              className={`flex-1 rounded-lg px-4 py-2.5 font-display text-sm font-medium transition sm:flex-none sm:px-8 ${
                mode === t.id
                  ? "bg-cyan-600 text-white shadow-lg shadow-cyan-900/30"
                  : "text-slate-400 hover:bg-slate-800/80 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {mode === "webcam" ? <WebcamCapture /> : <ImageUpload />}
      </main>

      <footer className="border-t border-surface-border py-6 text-center font-sans text-xs text-slate-600">
        FER · Flask + TensorFlow · React + Tailwind
      </footer>
    </div>
  );
}
