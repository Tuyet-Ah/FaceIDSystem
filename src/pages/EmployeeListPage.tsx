import { Camera, CameraOff, Eye, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import DeleteConfirmationModal from "../components/employee/DeleteConfirmationModal";
import useEmployees from "../hooks/useEmployees";
import type { Employee } from "../types";

export default function EmployeeListPage() {
  const navigate = useNavigate();
  const { employees, deleteEmployee, loading, error } = useEmployees();
  const [deleting, setDeleting] = useState<Employee | null>(null);

  return (
    <div className="page-shell">
      <div className="mb-11 flex items-center justify-between px-4">
        <h2 className="page-title">Employee information</h2>
        <button className="flex h-49 min-w-[292px] items-center justify-center gap-2 rounded-xl bg-white text-[24px] font-bold" onClick={() => navigate("/employees/new")}>
          <Plus size={22} strokeWidth={4} /> New employee
        </button>
      </div>
      <section className="min-h-[518px] rounded-[11px] bg-white px-11 py-[55px]">
        {error && <p className="mb-5 rounded-lg bg-red-50 p-4 font-semibold text-red-700">{error}</p>}
        <div className="overflow-x-auto rounded-[11px] border border-[#00233a]">
          <table className="w-full min-w-[760px] text-left text-[20px] font-semibold">
            <thead className="bg-[#00233a] text-[25px] text-white">
              <tr><th>ID</th><th>Name</th><th>Status</th><th className="text-center">View</th><th className="text-center">Delete</th></tr>
            </thead>
            <tbody>
              {employees.map((employee) => {
                const synced = employee.faceStatus === "READY";
                return (
                  <tr key={employee.id}>
                    <td>{employee.id}</td><td>{employee.name}</td>
                    <td><span className="inline-flex items-center gap-3">{synced ? "Ready" : employee.faceStatus}{synced ? <Camera className="text-[#44a6db]" fill="#44a6db" /> : <CameraOff className="text-[#ff4a4f]" />}</span></td>
                <td className="text-center"><button aria-label={`View ${employee.name}`} onClick={() => navigate(`/employees/${employee.id}`)}><Eye size={35} color="#009FE3" fill="white" /></button></td>
                    <td className="text-center"><button aria-label={`Delete ${employee.name}`} className="rounded-full bg-[#ff4141] p-[7px] text-white" onClick={() => setDeleting(employee)}><Trash2 size={24} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!loading && !employees.length && <p className="py-14 text-center text-lg text-slate-500">No employees found.</p>}
          {loading && <p className="py-14 text-center text-lg text-slate-500">Loading employees...</p>}
        </div>
      </section>
      {deleting && (
        <DeleteConfirmationModal
          employeeName={deleting.name}
          onCancel={() => setDeleting(null)}
          onConfirm={() => { void deleteEmployee(deleting.id).then(() => setDeleting(null)); }}
        />
      )}
    </div>
  );
}
