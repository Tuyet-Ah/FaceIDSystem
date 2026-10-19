import type { EmployeeFields } from "../../types";

export const emptyEmployee: EmployeeFields = {
  name: "",
  dateOfBirth: "",
  address: "",
  phone: "",
  email: "",
  department: "",
};
export const employeeFields: { key: keyof EmployeeFields; label: string }[] = [
  { key: "name", label: "Họ và tên" },
  { key: "dateOfBirth", label: "Ngày sinh (YYYY-MM-DD)" },
  { key: "address", label: "Địa chỉ" },
  { key: "phone", label: "Số điện thoại" },
  { key: "email", label: "Email" },
  { key: "department", label: "Phòng ban" },
];

export function validateEmployee(fields: EmployeeFields, imageCount: number, isNew: boolean) {
  const errors: Partial<Record<keyof EmployeeFields | "images", string>> = {};
  employeeFields.forEach(({ key }) => {
    if (!fields[key].trim()) errors[key] = "Thông tin bắt buộc";
  });
  if (fields.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(fields.email.trim()))
    errors.email = "Email không hợp lệ";
  if (fields.phone && !/^\+?[\d\s().-]{8,20}$/.test(fields.phone.trim()))
    errors.phone = "Số điện thoại không hợp lệ";
  const birthday = fields.dateOfBirth;
  const parsed = new Date(`${birthday}T00:00:00Z`);
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(birthday) ||
    Number.isNaN(parsed.getTime()) ||
    parsed.toISOString().slice(0, 10) !== birthday ||
    parsed > new Date()
  ) {
    errors.dateOfBirth = "Nhập ngày sinh hợp lệ, dạng YYYY-MM-DD";
  }
  if ((isNew || imageCount > 0) && (imageCount < 5 || imageCount > 10))
    errors.images = "Cần từ 5 đến 10 ảnh khuôn mặt";
  return errors;
}
