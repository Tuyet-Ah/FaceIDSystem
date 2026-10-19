import type { ImagePickerAsset } from "expo-image-picker";
import { Platform } from "react-native";
import { File } from "expo-file-system";
import type {
  Employee,
  EmployeeFields,
  RecognitionLog,
  RecognitionResult,
  SavedEmployee,
} from "../types";
import { ApiError, request } from "./client";

export type UploadImage = Pick<ImagePickerAsset, "uri" | "mimeType">;

export function appendImage(form: FormData, field: string, image: UploadImage) {
  // Expo SDK 57 fetch accepts File/Blob parts, not React Native URI descriptors.
  const file = new File(image.uri);
  form.append(field, file, file.name);
}

export const employeeService = {
  list: (server: string, signal?: AbortSignal) =>
    request<Employee[]>(server, "/employees", { signal }),
  get: (server: string, id: number, signal?: AbortSignal) =>
    request<Employee>(server, `/employees/${id}`, { signal }),
  save: (
    server: string,
    fields: EmployeeFields,
    images: ImagePickerAsset[],
    primary: number,
    id?: number,
  ) => {
    const body = new FormData();
    Object.entries(fields).forEach(([key, value]) => body.append(key, value.trim()));
    images.forEach((image) => appendImage(body, "images", image));
    if (images.length) body.append("primaryImageIndex", String(primary));
    return request<SavedEmployee>(server, id ? `/employees/${id}` : "/employees", {
      method: id ? "PUT" : "POST",
      body,
    });
  },
  remove: (server: string, id: number) =>
    request<void>(server, `/employees/${id}`, { method: "DELETE" }),
};

export const logService = {
  list: (
    server: string,
    signal?: AbortSignal,
    filters: { employeeId?: number; status?: RecognitionLog["status"] } = {},
  ) => {
    const query = new URLSearchParams();
    if (filters.employeeId !== undefined) query.set("employeeId", String(filters.employeeId));
    if (filters.status) query.set("status", filters.status);
    const suffix = query.toString();
    return request<RecognitionLog[]>(server, `/recognition-logs${suffix ? `?${suffix}` : ""}`, {
      signal,
    });
  },
  remove: (server: string, id: number) =>
    request<void>(server, `/recognition-logs/${id}`, { method: "DELETE" }),
};

export async function recognize(
  server: string,
  image: UploadImage,
  options: { signal?: AbortSignal; live?: boolean } = {},
): Promise<RecognitionResult> {
  const body = new FormData();
  appendImage(body, "image", image);
  body.append(
    "device",
    `${Platform.OS === "ios" ? "iOS" : "Android"}${options.live ? " Live Camera" : ""}`,
  );
  try {
    return await request<RecognitionResult>(server, "/recognitions", {
      method: "POST",
      body,
      signal: options.signal,
      timeout: options.live ? 15_000 : 120_000,
    });
  } catch (error) {
    // Validation failures are also recorded by the backend as UNKNOWN recognition logs.
    if (error instanceof ApiError && error.status === 422) {
      return { matched: false, status: "UNKNOWN", message: error.message };
    }
    throw error;
  }
}
