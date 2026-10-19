export interface Employee {
  id: number;
  name: string;
  dateOfBirth: string;
  address: string;
  phone: string;
  email: string;
  department: string;
  faceStatus: "PROCESSING" | "READY" | "FAILED";
  primaryPhotoUrl: string | null;
  faceImageCount: number;
}

export type EmployeeFields = Pick<
  Employee,
  "name" | "dateOfBirth" | "address" | "phone" | "email" | "department"
>;
export interface ImageWarning {
  index?: number;
  fileName: string;
  message: string;
}
export type SavedEmployee = Employee & { imageWarnings?: ImageWarning[] };

export interface RecognitionLog {
  id: number;
  employeeId: number | null;
  name: string;
  department: string;
  status: "SUCCESS" | "UNKNOWN";
  similarity: number | null;
  threshold?: number | null;
  imageUrl: string | null;
  recognizedAt: string;
  attendancePeriod: "MORNING" | "AFTERNOON" | null;
  reason: string | null;
  device: string;
}

export interface RecognitionResult {
  matched: boolean;
  status: "SUCCESS" | "UNKNOWN";
  employee?: Employee | null;
  similarity?: number;
  message: string;
  duplicate?: boolean;
  logId?: number;
}

export interface CameraStatus {
  state: "STARTING" | "RUNNING" | "STOPPED" | "ERROR";
  running: boolean;
  error: string | null;
  lastRecognition: {
    employee: Employee;
    recognizedAt: string;
    alreadyCheckedIn: boolean;
    similarity: number;
  } | null;
}
