import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api",
  timeout: 120_000,
});

export function mediaUrl(path: string | null | undefined) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  const apiBase = import.meta.env.VITE_API_URL ?? "/api";
  if (apiBase.startsWith("http")) return new URL(path, apiBase).toString();
  return path;
}

export default api;
