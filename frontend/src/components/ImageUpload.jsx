import React, { useCallback, useState } from "react";
import { predictImage } from "../api";
import EmotionResult from "./EmotionResult";

/**
 * Drag-and-drop or click upload; preview and run prediction.
 */
export default function ImageUpload() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runPredict = useCallback(async (blob) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImage(blob);
      setResult(data);
    } catch (e) {
      const msg =
        e.response?.data?.message ||
        e.message ||
        "Could not reach the API. Check REACT_APP_API_URL and CORS.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const onFile = useCallback(
    (f) => {
      if (!f || !f.type.startsWith("image/")) {
        setError("Please choose a JPEG or PNG image.");
        return;
      }
      setFile(f);
      setError(null);
      setResult(null);
      const url = URL.createObjectURL(f);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      runPredict(f);
    },
    [runPredict]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      const f = e.dataTransfer?.files?.[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onInputChange = (e) => {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div className="space-y-6">
      <div
        role="button"
        tabIndex={0}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            document.getElementById("file-input")?.click();
          }
        }}
        className="group relative cursor-pointer rounded-2xl border-2 border-dashed border-surface-border bg-surface-card/50 p-10 text-center transition hover:border-cyan-500/50 hover:bg-surface-card"
      >
        <input
          id="file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={onInputChange}
        />
        <label htmlFor="file-input" className="cursor-pointer">
          <p className="font-display text-lg text-slate-200">
            Drop an image here or{" "}
            <span className="text-cyan-400 underline decoration-cyan-400/40">
              browse
            </span>
          </p>
          <p className="mt-2 font-sans text-sm text-slate-500">
            JPEG / PNG — face should be visible
          </p>
        </label>
      </div>

      {preview && (
        <div className="overflow-hidden rounded-2xl border border-surface-border bg-black/40">
          <img
            src={preview}
            alt="Upload preview"
            className="mx-auto max-h-80 w-auto object-contain"
          />
        </div>
      )}

      <EmotionResult
        result={result}
        loading={loading}
        error={error}
        emptyMessage={
          file ? undefined : "Upload a photo to see expression probabilities."
        }
      />
    </div>
  );
}
