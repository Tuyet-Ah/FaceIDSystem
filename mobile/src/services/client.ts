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
  let rejectAbort: (reason: Error) => void = () => {};
  const aborted = new Promise<never>((_, reject) => {
    rejectAbort = reject;
  });
  const cancel = () => {
    controller.abort();
    rejectAbort(
      new Error(
        options.signal?.aborted
          ? "Yêu cầu đã được hủy."
          : "Máy chủ phản hồi quá lâu. Vui lòng thử lại.",
      ),
    );
  };
  options.signal?.addEventListener("abort", cancel);
  const timer = setTimeout(cancel, options.timeout ?? (options.body ? 120_000 : 15_000));
  try {
    if (options.signal?.aborted) cancel();
    const response = await Promise.race([
      fetch(`${server}/api${path}`, {
        method: options.method ?? "GET",
        body: options.body,
        signal: controller.signal,
        headers: { Accept: "application/json" },
      }),
      aborted,
    ]);
    if (response.status === 204) return undefined as T;
    // Bound body reads too: some native transports can stall after receiving headers.
    const bodyTimer = setTimeout(cancel, 15_000);
    let text: string;
    try {
      text = await Promise.race([response.text(), aborted]);
    } catch (error) {
      if (!response.ok && !options.signal?.aborted) {
        throw new ApiError(
          `Máy chủ từ chối yêu cầu (HTTP ${response.status}), nhưng không đọc được chi tiết lỗi.`,
          response.status,
        );
      }
      throw error;
    } finally {
      clearTimeout(bodyTimer);
    }
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
    if (error instanceof ApiError) throw error;
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
