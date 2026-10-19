import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "../layout/AppLayout";
import EmployeeDetailPage from "../pages/EmployeeDetailPage";
import EmployeeListPage from "../pages/EmployeeListPage";
import EmployeeRegistrationPage from "../pages/EmployeeRegistrationPage";
import SystemLogsPage from "../pages/SystemLogsPage";
import AttendancePage from "../pages/AttendancePage";

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/employees" replace />} />
        <Route path="employees" element={<EmployeeListPage />} />
        <Route path="employees/new" element={<EmployeeRegistrationPage />} />
        <Route path="employees/:id" element={<EmployeeDetailPage />} />
        <Route path="employees/:id/edit" element={<EmployeeRegistrationPage />} />
        <Route path="recognition" element={<AttendancePage />} />
        <Route path="logs" element={<SystemLogsPage />} />
        <Route path="*" element={<Navigate to="/employees" replace />} />
      </Route>
    </Routes>
  );
}
