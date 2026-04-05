import React, { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { predictImage } from "../api";
import EmotionResult from "./EmotionResult";

const CAPTURE_MS = 2000;

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

/**
 * Periodic webcam capture, API prediction, and canvas face overlay.
 */
export default function WebcamCapture() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [camReady, setCamReady] = useState(false);

  const drawBox = useCallback((faceBox, video) => {
    const canvas = canvasRef.current;
    const wrap = containerRef.current;
    if (!canvas || !wrap || !video || !faceBox) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const rect = wrap.getBoundingClientRect();
    const dispW = rect.width;
    const dispH = (vh / vw) * dispW;

    canvas.width = dispW;
    canvas.height = dispH;

    const sx = dispW / vw;
    const sy = dispH / vh;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.strokeStyle = "rgba(34, 211, 238, 0.95)";
    ctx.lineWidth = 3;
    ctx.strokeRect(
      faceBox.x * sx,
      faceBox.y * sy,
      faceBox.width * sx,
      faceBox.height * sy
    );
  }, []);

  useEffect(() => {
    const video = webcamRef.current?.video;
    if (result?.face_detected && result?.face_box && video) {
      drawBox(result.face_box, video);
    } else {
      drawBox(null, video);
    }
  }, [result, drawBox, camReady]);

  useEffect(() => {
    const captureAndPredict = () => {
      const shot = webcamRef.current?.getScreenshot?.();
      if (!shot) return;

      fetch(shot)
        .then((r) => r.blob())
        .then((blob) => {
          setLoading(true);
          setError(null);
          return predictImage(blob);
        })
        .then((data) => {
          setResult(data);
        })
        .catch((e) => {
          const msg =
            e.response?.data?.message ||
            e.message ||
            "Prediction failed. Is the backend running?";
          setError(msg);
        })
        .finally(() => setLoading(false));
    };

    const t0 = setTimeout(captureAndPredict, 800);
    const id = setInterval(captureAndPredict, CAPTURE_MS);
    return () => {
      clearTimeout(t0);
      clearInterval(id);
    };
  }, []);

  const onUserMedia = useCallback(() => {
    setCamReady(true);
  }, []);

  return (
    <div className="space-y-6">
      <div
        ref={containerRef}
        className="relative mx-auto w-full max-w-2xl overflow-hidden rounded-2xl border border-surface-border bg-black shadow-lg"
      >
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          screenshotQuality={0.92}
          videoConstraints={videoConstraints}
          onUserMedia={onUserMedia}
          className="block w-full"
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute left-0 top-0 h-full w-full"
          aria-hidden
        />
        <div className="absolute bottom-3 left-3 rounded-lg bg-black/60 px-3 py-1 font-sans text-xs text-cyan-300">
          Snapshot every {CAPTURE_MS / 1000}s
        </div>
      </div>

      <EmotionResult
        result={result}
        loading={loading}
        error={error}
        emptyMessage="Waiting for the first capture…"
      />
    </div>
  );
}
