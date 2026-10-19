import { Camera, CameraOff, CheckCircle2, Clock3, UsersRound } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import useEmployees from "../hooks/useEmployees";
import api, { mediaUrl } from "../services/api";
import type { Employee } from "../types";

function getLogPeriod(log: { recognizedAt: string; attendancePeriod?: "MORNING" | "AFTERNOON" | null }) {
  return log.attendancePeriod ?? (new Date(log.recognizedAt).getHours() < 12 ? "MORNING" : "AFTERNOON");
}

interface CameraStatus {
  state: "STARTING" | "RUNNING" | "STOPPED" | "ERROR";
  running: boolean;
  cameraIndex: number;
  error: string | null;
  lastRecognition: {
    employee: Employee;
    recognizedAt: string;
    similarity: number;
    alreadyCheckedIn: boolean;
    attendancePeriod: "MORNING" | "AFTERNOON";
    logId: number;
  } | null;
}

export default function AttendancePage() {
  const { employees, logs } = useEmployees();
  const [camera, setCamera] = useState<CameraStatus | null>(null);
  const [connectionError, setConnectionError] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const response = await api.get<CameraStatus>("/camera/status");
        if (active) {
          setCamera(response.data);
          setConnectionError(false);
        }
      } catch {
        if (active) setConnectionError(true);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const today = new Date().toLocaleDateString("en-CA");
  const checkedIn = useMemo(() => {
    const unique = new Map<string, (typeof logs)[number]>();
    logs.forEach((log) => {
      if (log.status !== "SUCCESS" || !log.employeeId) return;
      const recognizedAt = new Date(log.recognizedAt);
      const date = recognizedAt.toLocaleDateString("en-CA");
      const period = getLogPeriod(log);
      const key = `${log.employeeId}-${period}`;
      if (date === today && !unique.has(key)) unique.set(key, log);
    });
    return [...unique.values()];
  }, [logs, today]);

  const latestLog = logs.find((log) => log.status === "SUCCESS");
  const latestEmployee = employees.find((item) => item.id === latestLog?.employeeId);
  const latestTime = latestLog?.recognizedAt;
  const latestSimilarity = latestLog?.similarity;
  const latestImage = latestEmployee?.primaryPhotoUrl ?? latestLog?.imageUrl;
  const cameraRunning = camera?.state === "RUNNING";

  return (
    <div className="page-shell">
      <h2 className="page-title mb-8 px-4">Attendance status</h2>

      <div className="mb-6 grid gap-5 md:grid-cols-3">
        <section className="rounded-xl bg-white p-5">
          <div className="flex items-center gap-3 text-[#3677b4]"><Camera size={28} /><h3 className="text-lg font-bold">Local camera</h3></div>
          <p className={`mt-3 text-2xl font-bold ${cameraRunning ? "text-emerald-600" : "text-red-600"}`}>
            {connectionError ? "BACKEND OFFLINE" : camera?.state ?? "CONNECTING"}
          </p>
          <p className="mt-1 text-sm text-slate-500">Camera index: {camera?.cameraIndex ?? 0}</p>
        </section>
        <section className="rounded-xl bg-white p-5">
          <div className="flex items-center gap-3 text-[#3677b4]"><UsersRound size={28} /><h3 className="text-lg font-bold">Ready employees</h3></div>
          <p className="mt-3 text-3xl font-bold">{employees.filter((item) => item.faceStatus === "READY").length}</p>
        </section>
        <section className="rounded-xl bg-white p-5">
          <div className="flex items-center gap-3 text-[#3677b4]"><CheckCircle2 size={28} /><h3 className="text-lg font-bold">Check-ins today</h3></div>
          <p className="mt-3 text-3xl font-bold text-emerald-600">{checkedIn.length}</p>
        </section>
      </div>

      {(camera?.error || connectionError) && (
        <div className="mb-6 flex items-center gap-3 rounded-xl border border-red-300 bg-red-50 p-4 font-semibold text-red-700">
          <CameraOff /> {connectionError ? "Không kết nối được backend FaceID." : camera?.error}
        </div>
      )}

      <section className="grid min-h-[440px] gap-7 rounded-[11px] bg-white p-4 sm:p-6 lg:p-8 lg:grid-cols-[280px_1fr] xl:grid-cols-[310px_1fr]">
        <div className="rounded-xl bg-[#eef7fc] p-5 text-center">
          <h3 className="mb-5 text-xl font-bold text-[#174b6e]">Latest recognition</h3>
          {latestLog ? (
            <>
              <img src={mediaUrl(latestImage) || "/assets/employee-portrait.jpg"} alt={latestLog.name} className="mx-auto h-64 w-52 rounded-xl object-cover" />
              <p className="mt-4 text-2xl font-bold text-emerald-700">{latestLog.name}</p>
              <p className="font-semibold text-slate-600">{latestLog.department}</p>
              <p className="mx-auto mt-2 w-fit rounded-full bg-sky-100 px-3 py-1 text-sm font-bold text-sky-700">
                {getLogPeriod(latestLog) === "MORNING" ? "Morning shift" : "Afternoon shift"}
              </p>
              {latestTime && <p className="mt-2 flex items-center justify-center gap-2 text-sm"><Clock3 size={16} />{new Date(latestTime).toLocaleString()}</p>}
              {latestSimilarity != null && <p className="mt-1 text-sm font-bold">Confidence: {(latestSimilarity * 100).toFixed(1)}%</p>}
            </>
          ) : (
            <div className="grid h-72 place-items-center text-slate-400"><div><Camera size={70} className="mx-auto" /><p className="mt-3">Waiting for recognition...</p></div></div>
          )}
        </div>

        <div>
          <h3 className="mb-5 text-2xl font-bold text-[#174b6e]">Morning and afternoon check-ins</h3>
          <div className="w-full overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full table-fixed text-left text-sm">
              <colgroup>
                <col className="w-[28%]" />
                <col className="w-[17%]" />
                <col className="w-[18%]" />
                <col className="w-[16%]" />
                <col className="w-[21%]" />
              </colgroup>
              <thead className="bg-[#174b6e] text-white">
                <tr>
                  <th className="px-3 py-3 font-semibold">Name</th>
                  <th className="px-3 py-3 font-semibold">Department</th>
                  <th className="px-3 py-3 font-semibold">Shift</th>
                  <th className="px-3 py-3 font-semibold">Time</th>
                  <th className="px-3 py-3 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {checkedIn.map((log) => (
                  <tr key={`${log.employeeId}-${getLogPeriod(log)}`} className="border-b border-slate-100">
                    <td className="max-w-0 truncate px-3 py-3 font-bold" title={log.name ?? ""}>{log.name}</td>
                    <td className="max-w-0 truncate px-3 py-3" title={log.department ?? ""}>{log.department}</td>
                    <td className="px-3 py-3">
                      <span className="inline-block rounded-full bg-sky-100 px-2 py-0.5 text-xs font-bold text-sky-700">
                        {getLogPeriod(log) === "AFTERNOON" ? "Afternoon" : "Morning"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">{new Date(log.recognizedAt).toLocaleTimeString()}</td>
                    <td className="px-3 py-3">
                      <span className="inline-block rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-bold text-emerald-700">Checked in</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!checkedIn.length && <p className="py-16 text-center text-slate-500">No employee has checked in today.</p>}
          </div>
        </div>
      </section>
    </div>
  );
}
