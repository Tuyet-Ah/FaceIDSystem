import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { employeeService } from "../services/employeeService";
import { logService } from "../services/logService";
import type { Employee, EmployeeInput, RecognitionLog } from "../types";

interface EmployeeContextValue {
  employees: Employee[];
  logs: RecognitionLog[];
  loading: boolean;
  error: string | null;
  recognitionNotice: RecognitionLog | null;
  addEmployee: (input: EmployeeInput) => Promise<Employee>;
  updateEmployee: (id: number, input: EmployeeInput) => Promise<Employee>;
  deleteEmployee: (id: number) => Promise<void>;
  deleteLog: (id: number) => Promise<void>;
  getEmployee: (id: number) => Employee | undefined;
  refreshEmployees: () => Promise<void>;
  refreshLogs: () => Promise<void>;
  dismissRecognitionNotice: () => void;
}

export const EmployeeContext = createContext<EmployeeContextValue | null>(null);

export function EmployeeProvider({ children }: { children: ReactNode }) {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [logs, setLogs] = useState<RecognitionLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [employeeError, setEmployeeError] = useState<string | null>(null);
  const [logError, setLogError] = useState<string | null>(null);
  const [recognitionNotice, setRecognitionNotice] = useState<RecognitionLog | null>(null);
  const logsInitialized = useRef(false);
  const latestSuccessId = useRef<number | null>(null);

  const refreshEmployees = useCallback(async () => {
    try {
      const response = await employeeService.getAll();
      setEmployees(response.data);
      setEmployeeError(null);
    } catch {
      setEmployeeError("Không thể kết nối tới backend. Hãy kiểm tra server Flask.");
    }
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const response = await logService.getAll();
      const newestSuccess = response.data.find((log) => log.status === "SUCCESS");
      if (!logsInitialized.current) {
        latestSuccessId.current = newestSuccess?.id ?? null;
        logsInitialized.current = true;
      } else if (
        newestSuccess
        && (latestSuccessId.current == null || newestSuccess.id > latestSuccessId.current)
      ) {
        latestSuccessId.current = newestSuccess.id;
        setRecognitionNotice(newestSuccess);
      }
      setLogs(response.data);
      setLogError(null);
    } catch {
      setLogError("Không thể tải lịch sử nhận diện.");
    }
  }, []);

  useEffect(() => {
    Promise.all([refreshEmployees(), refreshLogs()]).finally(() => setLoading(false));
  }, [refreshEmployees, refreshLogs]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshEmployees();
      void refreshLogs();
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [refreshEmployees, refreshLogs]);

  const error = employeeError ?? logError;

  const dismissRecognitionNotice = useCallback(() => {
    setRecognitionNotice(null);
  }, []);

  const addEmployee = useCallback(async (input: EmployeeInput) => {
    const response = await employeeService.create(input);
    setEmployees((current) => [response.data, ...current]);
    return response.data;
  }, []);

  const updateEmployee = useCallback(async (id: number, input: EmployeeInput) => {
    const response = await employeeService.update(id, input);
    setEmployees((current) => current.map((item) => item.id === id ? response.data : item));
    return response.data;
  }, []);

  const deleteEmployee = useCallback(async (id: number) => {
    await employeeService.remove(id);
    setEmployees((current) => current.filter((employee) => employee.id !== id));
  }, []);

  const deleteLog = useCallback(async (id: number) => {
    await logService.remove(id);
    setLogs((current) => current.filter((log) => log.id !== id));
  }, []);

  const getEmployee = useCallback(
    (id: number) => employees.find((employee) => employee.id === id),
    [employees],
  );

  const value = useMemo(
    () => ({
      employees, logs, loading, error, recognitionNotice, addEmployee, updateEmployee,
      deleteEmployee, deleteLog, getEmployee, refreshEmployees, refreshLogs,
      dismissRecognitionNotice,
    }),
    [
      employees, logs, loading, error, recognitionNotice, addEmployee, updateEmployee,
      deleteEmployee, deleteLog, getEmployee, refreshEmployees, refreshLogs,
      dismissRecognitionNotice,
    ],
  );

  return <EmployeeContext.Provider value={value}>{children}</EmployeeContext.Provider>;
}
