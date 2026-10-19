import api from "./api";
import type { RecognitionResult } from "../types";

export const recognitionService = {
  recognize: (image: File, device = "Browser Camera") => {
    const formData = new FormData();
    formData.append("image", image);
    formData.append("device", device);
    return api.post<RecognitionResult>("/recognitions", formData);
  },
};
