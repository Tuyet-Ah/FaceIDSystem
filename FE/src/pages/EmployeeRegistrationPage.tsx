import { Navigate, useNavigate, useParams } from "react-router-dom";
import EmployeeForm from "../components/employee/EmployeeForm";
import useEmployees from "../hooks/useEmployees";

export default function EmployeeRegistrationPage() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { addEmployee, updateEmployee, getEmployee, loading } = useEmployees();
  const editing = id !== undefined;
  const employee = editing ? getEmployee(Number(id)) : undefined;
  if (loading) return <div className="page-shell text-xl text-white">Loading...</div>;
  if (editing && !employee) return <Navigate to="/employees" replace />;

  return (
    <div className="page-shell">
      <h2 className="page-title mb-10 px-4">{editing ? "Update employee information" : "Employee information registration"}</h2>
      <EmployeeForm
        employee={employee}
        onCancel={() => navigate(editing ? `/employees/${id}` : "/employees")}
        onSave={async (value) => {
          if (editing) {
            await updateEmployee(Number(id), value);
            navigate(`/employees/${id}`);
          } else {
            const created = await addEmployee(value);
            navigate(`/employees/${created.id}`);
          }
        }}
      />
    </div>
  );
}
