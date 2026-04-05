import React from "react";

const EMOJI = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😄",
  neutral: "😐",
  sad: "😢",
  surprise: "😮",
};

const BAR_COLORS = {
  angry: "bg-red-500",
  disgust: "bg-lime-600",
  fear: "bg-purple-500",
  happy: "bg-yellow-400",
  neutral: "bg-slate-400",
  sad: "bg-blue-500",
  surprise: "bg-orange-400",
};

/** Matches backend label_map.json / FER2013 class indices 0–6 */
const ORDER = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "sad",
  "surprise",
  "neutral",
];

/**
 * Dominant emotion, emoji, confidence %, and animated probability bars.
 */
export default function EmotionResult({
  result,
  loading,
  error,
  emptyMessage,
}) {
  if (loading) {
    return (
      <div className="rounded-2xl border border-surface-border bg-surface-card p-8 text-center">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
        <p className="mt-4 font-sans text-slate-400">Analyzing…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-900/50 bg-red-950/30 p-6">
        <p className="font-display text-lg text-red-300">Request failed</p>
        <p className="mt-2 font-sans text-sm text-red-200/80">{error}</p>
      </div>
    );
  }

  if (!result || result.face_detected === false) {
    return (
      <div className="rounded-2xl border border-surface-border bg-surface-card p-8 text-center">
        <p className="font-sans text-slate-400">
          {emptyMessage || "No face detected or no prediction yet."}
        </p>
      </div>
    );
  }

  if (result.emotion == null && result.error) {
    return (
      <div className="rounded-2xl border border-amber-900/50 bg-amber-950/20 p-6">
        <p className="font-sans text-amber-200/90">{result.message || result.error}</p>
      </div>
    );
  }

  const emotion = result.emotion;
  const conf = typeof result.confidence === "number" ? result.confidence : 0;
  const probs = result.all_probabilities || {};

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card p-6 shadow-xl shadow-black/20">
      <div className="flex flex-col items-center gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div className="text-center sm:text-left">
          <p className="font-display text-sm font-medium uppercase tracking-widest text-slate-500">
            Dominant expression
          </p>
          <div className="mt-1 flex flex-wrap items-center justify-center gap-3 sm:justify-start">
            <span className="text-5xl leading-none" aria-hidden>
              {EMOJI[emotion] || "❔"}
            </span>
            <span className="font-display text-3xl font-semibold capitalize text-white">
              {emotion}
            </span>
          </div>
        </div>
        <div className="rounded-xl bg-surface px-4 py-2 text-center">
          <p className="font-sans text-xs uppercase text-slate-500">Confidence</p>
          <p className="font-display text-2xl font-bold text-cyan-400">
            {(conf * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="mt-8 space-y-3">
        <p className="font-display text-sm font-medium text-slate-400">
          All probabilities
        </p>
        {ORDER.map((key) => {
          const p = typeof probs[key] === "number" ? probs[key] : 0;
          const pct = Math.round(p * 1000) / 10;
          const color = BAR_COLORS[key] || "bg-slate-500";
          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between font-sans text-xs text-slate-400">
                <span className="capitalize">{key}</span>
                <span>{pct}%</span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${color}`}
                  style={{ width: `${Math.min(100, pct)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
