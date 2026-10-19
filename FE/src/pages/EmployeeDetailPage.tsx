import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import useEmployees from "../hooks/useEmployees";
import { mediaUrl } from "../services/api";

export default function EmployeeDetailPage() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { getEmployee, loading } = useEmployees();
  const employee = getEmployee(Number(id));
  if (loading) return <div className="page-shell text-xl text-white">Loading...</div>;
  if (!employee) return <Navigate to="/employees" replace />;
  const synced = employee.faceStatus === "READY";
  const details = [
    ["ID", employee.id], ["Name", employee.name], ["Date of birth", employee.dateOfBirth],
    ["Address", employee.address], ["Phone", employee.phone], ["Email", employee.email],
    ["Department", employee.department], ["Face images", employee.faceImageCount],
  ];

  return (
    <div className="page-shell">
      <h2 className="page-title mb-6 px-4">Employee information</h2>
      <section className="grid min-h-[540px] grid-cols-[1fr_330px] gap-10 rounded-[11px] bg-white px-[92px] py-[30px]">
        <dl className="space-y-[18px] text-[20px] font-bold">
          {details.map(([label, value]) => <div className="grid grid-cols-[178px_1fr]" key={label}><dt className="text-[26px] text-[#3677b4]">{label}</dt><dd className="pt-1">{value}</dd></div>)}
          <div className="grid grid-cols-[178px_1fr]">
            <dt className="text-[26px] text-[#3677b4]">Status</dt>
            <dd className={`flex items-center gap-5 pt-1 underline ${synced ? "text-emerald-600" : "text-red-600"}`}>{employee.faceStatus}{synced ? <CheckCircle2 /> : <AlertTriangle />}</dd>
          </div>
          <div className="grid grid-cols-[178px_1fr]">
            <dt className="text-[26px] leading-7 text-[#3677b4]">Attendance<br />history</dt>
            <dd className="pt-2"><button className="underline" onClick={() => navigate(`/logs?employee=${employee.id}`)}>View Detail</button></dd>
          </div>
        </dl>
        <div className="flex flex-col justify-between">
          <figure className="self-end">
            <div className="rounded-2xl border-4 border-[#4f8dcc] bg-[#eaf5fb] p-2 shadow-[0_12px_30px_rgba(0,35,58,0.18)]">
              <img src={mediaUrl(employee.primaryPhotoUrl) || "/assets/employee-portrait.jpg"} alt={employee.name} className="h-[302px] w-[224px] rounded-xl object-cover object-top" />
            </div>
            <figcaption className="mt-3 text-center text-sm font-bold text-[#3677b4]">Profile photo</figcaption>
          </figure>
          <div className="flex justify-end gap-12">
            <button className="action-button" onClick={() => navigate("/employees")}>BACK</button>
            <button className="action-button" onClick={() => navigate(`/employees/${employee.id}/edit`)}>UPDATE</button>
          </div>
        </div>
      </section>
    </div>
  );
}
