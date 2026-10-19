import api from "./api";
import type { RecognitionLog } from "../types";

export const logService = {
  getAll: () => api.get<RecognitionLog[]>("/recognition-logs"),
  getById: (id: number) => api.get<RecognitionLog>(`/recognition-logs/${id}`),
  remove: (id: number) => api.delete(`/recognition-logs/${id}`),
};
