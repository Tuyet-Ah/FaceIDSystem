import { NavLink, Outlet } from "react-router-dom";
import { DatabaseZap, ScanFace, UsersRound, UserRound } from "lucide-react";
import RecognitionSuccessToast from "../components/logs/RecognitionSuccessToast";

export default function AppLayout() {
  return (
    <div className="min-h-screen bg-[#00233a] text-[#071f36]">
      <RecognitionSuccessToast />
      <header className="flex h-[85px] items-center justify-between bg-white px-8">
        <div className="flex items-center gap-7 text-[#4f8dcc]">
          <ScanFace size={72} strokeWidth={1.8} />
          <h1 className="text-[36px] font-bold tracking-[-1px]">
            FaceID Management System
          </h1>
        </div>
        <div className="flex items-center gap-7 text-[18px] text-[#2b5595]">
          <div className="leading-[22px]">
            <p><strong>Mode:</strong> Local system</p>
            <p>No authentication</p>
          </div>
          <div className="grid size-50 place-items-center rounded-full border border-[#2e69b7] bg-[#eaf7ff]">
            <UserRound size={37} fill="#34aaf0" color="#34aaf0" />
          </div>
        </div>
      </header>

      <div className="flex min-h-[calc(100vh-85px)]">
        <aside className="w-[296px] shrink-0 bg-white px-[10px] py-[22px]">
          <nav className="space-y-[21px]">
            <NavLink to="/employees" className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}>
              <UsersRound size={34} color="currentColor" />
              Employee List
            </NavLink>
            <NavLink to="/recognition" className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}>
              <ScanFace size={34} color="currentColor" />
              Attendance
            </NavLink>
            <NavLink to="/logs" className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}>
              <DatabaseZap size={34} color="currentColor" />
              System Logs
            </NavLink>
          </nav>
        </aside>
        <main className="min-w-0 flex-1 bg-[#00233a]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
