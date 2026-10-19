import api from "./api";
import type { Employee, EmployeeInput } from "../types";

function toFormData(payload: EmployeeInput) {
  const formData = new FormData();
  formData.append("name", payload.name);
  formData.append("dateOfBirth", payload.dateOfBirth);
  formData.append("address", payload.address);
  formData.append("phone", payload.phone);
  formData.append("email", payload.email);
  formData.append("department", payload.department);
  payload.images.forEach((image) => formData.append("images", image));
  if (payload.images.length && payload.primaryImageIndex != null) {
    formData.append("primaryImageIndex", String(payload.primaryImageIndex));
  }
  return formData;
}

export const employeeService = {
  getAll: () => api.get<Employee[]>("/employees"),
  getById: (id: number) => api.get<Employee>(`/employees/${id}`),
  create: (payload: EmployeeInput) =>
    api.post<Employee>("/employees", toFormData(payload)),
  update: (id: number, payload: EmployeeInput) =>
    api.put<Employee>(`/employees/${id}`, toFormData(payload)),
  remove: (id: number) => api.delete(`/employees/${id}`),
};
