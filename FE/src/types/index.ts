export type FaceStatus = "PROCESSING" | "READY" | "FAILED";
export type RecognitionStatus = "SUCCESS" | "UNKNOWN";

export interface Employee {
  id: number;
  name: string;
  dateOfBirth: string;
  address: string;
  phone: string;
  email: string;
  department: string;
  faceStatus: FaceStatus;
  primaryPhotoUrl: string | null;
  faceImageCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface EmployeeInput {
  name: string;
  dateOfBirth: string;
  address: string;
  phone: string;
  email: string;
  department: string;
  images: File[];
  primaryImageIndex: number | null;
}

export interface RecognitionLog {
  id: number;
  employeeId: number | null;
  name: string;
  department: string;
  status: RecognitionStatus;
  similarity: number | null;
  threshold: number | null;
  reason: string | null;
  device: string;
  imageUrl: string | null;
  recognizedAt: string;
  attendancePeriod: "MORNING" | "AFTERNOON" | null;
}

export interface RecognitionResult {
  matched: boolean;
  status: RecognitionStatus;
  similarity?: number;
  threshold?: number;
  employee?: Employee | null;
  recognizedAt?: string;
  logId: number;
  imageUrl?: string | null;
  message: string;
  duplicate?: boolean;
}

export interface ApiErrorBody {
  message?: string;
  imageErrors?: Array<{ index: number; fileName: string; message: string }>;
}
