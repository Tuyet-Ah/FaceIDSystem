import type { ImageWarning } from "../types";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public imageErrors: ImageWarning[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Không thể thực hiện yêu cầu. Vui lòng thử lại.";
}

export function normalizeServer(value: string) {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    throw new Error("Địa chỉ máy chủ không hợp lệ");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("Địa chỉ máy chủ không hợp lệ");
  }
  if (!["/", "/api", "/api/"].includes(url.pathname)) {
    throw new Error("Nhập địa chỉ gốc, ví dụ http://192.168.1.10:5000");
  }
  return url.origin;
}

export function mediaUrl(server: string, path: string | null) {
  if (!path) return undefined;
  try {
    return new URL(path, `${server}/`).toString();
  } catch {
    return undefined;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: FormData;
  signal?: AbortSignal;
  timeout?: number;
}

export async function request<T>(
  server: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  options.signal?.addEventListener("abort", cancel);
  const timer = setTimeout(cancel, options.timeout ?? (options.body ? 120_000 : 15_000));
  try {
    const response = await fetch(`${server}/api${path}`, {
      method: options.method ?? "GET",
      body: options.body,
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new ApiError(
        `Máy chủ trả về dữ liệu không hợp lệ (HTTP ${response.status})`,
        response.status,
      );
    }
    if (!response.ok)
      throw new ApiError(
        data.message || `Yêu cầu thất bại (HTTP ${response.status})`,
        response.status,
        data.imageErrors ?? [],
      );
    return data as T;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    if (controller.signal.aborted) throw new Error("Máy chủ phản hồi quá lâu. Vui lòng thử lại.");
    if (error instanceof TypeError)
      throw new Error("Không kết nối được máy chủ. Kiểm tra Wi-Fi và địa chỉ máy chủ.");
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", cancel);
  }
}
