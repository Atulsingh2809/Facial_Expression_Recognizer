/**
 * Axios client for the FER Flask API (base URL from REACT_APP_API_URL).
 */

import axios from "axios";

const baseURL =
  process.env.REACT_APP_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:5000";

const client = axios.create({
  baseURL,
  timeout: 60000,
  headers: {},
});

/**
 * POST /predict with image file; returns parsed JSON or throws.
 * @param {Blob|File} imageBlob
 * @param {AbortSignal} [signal]
 */
export async function predictImage(imageBlob, signal) {
  const formData = new FormData();
  formData.append("image", imageBlob, "capture.jpg");

  const { data } = await client.post("/predict", formData, {
    signal,
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * GET /health
 */
export async function checkHealth() {
  const { data } = await client.get("/health");
  return data;
}

export { baseURL };
