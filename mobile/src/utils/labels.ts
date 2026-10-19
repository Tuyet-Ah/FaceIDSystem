import type { Employee, RecognitionLog } from "../types";

export const faceStatusLabel: Record<Employee["faceStatus"], string> = {
  READY: "Sẵn sàng nhận diện",
  PROCESSING: "Đang xử lý",
  FAILED: "Lỗi dữ liệu khuôn mặt",
};

export const recognitionStatusLabel: Record<RecognitionLog["status"], string> = {
  SUCCESS: "Thành công",
  UNKNOWN: "Không nhận diện được",
};

export function personName(name: string) {
  return name === "Unknown" ? "Chưa xác định" : name;
}

export function deviceLabel(device: string) {
  return device
    .replace("Live Camera", "- camera trực tiếp")
    .replace("Browser Camera", "Camera trình duyệt")
    .replace("Local Camera", "Camera máy chủ");
}
