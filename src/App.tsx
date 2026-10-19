import { BrowserRouter } from "react-router-dom";
import { EmployeeProvider } from "./context/EmployeeContext";
import AppRoutes from "./routes/AppRoutes";

export default function App() {
  return (
    <BrowserRouter>
      <EmployeeProvider>
        <AppRoutes />
      </EmployeeProvider>
    </BrowserRouter>
  );
}
