import { FileSearch, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import CheckLogDetailModal from "../components/logs/CheckLogDetailModal";
import DeleteLogConfirmationModal from "../components/logs/DeleteLogConfirmationModal";
import useEmployees from "../hooks/useEmployees";
import type { RecognitionLog } from "../types";

type TimeRange = "all" | "yesterday" | "last3days" | "last7days";

function matchesTimeRange(recognizedAt: string, range: TimeRange) {
  if (range === "all") return true;
  const logDate = new Date(recognizedAt);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  if (range === "yesterday") {
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    return logDate >= yesterday && logDate < today;
  }

  const start = new Date(today);
  start.setDate(start.getDate() - (range === "last3days" ? 2 : 6));
  return logDate >= start && logDate < tomorrow;
}

export default function SystemLogsPage() {
  const { logs, deleteLog } = useEmployees();
  const [params] = useSearchParams();
  const [query, setQuery] = useState(params.get("employee") ?? "");
  const [status, setStatus] = useState("all");
  const [timeRange, setTimeRange] = useState<TimeRange>("all");
  const [selected, setSelected] = useState<RecognitionLog | null>(null);
  const [pendingDelete, setPendingDelete] = useState<RecognitionLog | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const filtered = useMemo(() => logs.filter((log) => {
    const term = query.toLowerCase().trim();
    const queryMatch = !term || String(log.employeeId).includes(term) || log.name.toLowerCase().includes(term);
    return queryMatch && (status === "all" || log.status === status) && matchesTimeRange(log.recognizedAt, timeRange);
  }), [logs, query, status, timeRange]);

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteLog(pendingDelete.id);
      if (selected?.id === pendingDelete.id) setSelected(null);
      setPendingDelete(null);
    } catch (error: any) {
      setDeleteError(error?.response?.data?.message || "Unable to delete this system log.");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="px-[22px] py-[53px]">
      <div className="mb-11 grid grid-cols-[minmax(300px,1fr)_258px_236px] gap-16">
        <label className="filter-control"><Search size={31} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Enter ID/name" /></label>
        <select className="filter-control" value={status} onChange={(event) => setStatus(event.target.value)}>
          <option value="all">All Status --</option><option>SUCCESS</option><option>UNKNOWN</option>
        </select>
        <select className="filter-control" value={timeRange} onChange={(event) => setTimeRange(event.target.value as TimeRange)}>
          <option value="all">All time</option>
          <option value="yesterday">Yesterday</option>
          <option value="last3days">Last 3 days</option>
          <option value="last7days">Last 7 days</option>
        </select>
      </div>
      <div className="overflow-x-auto rounded-[11px] bg-[#174b6e] pb-7">
        <table className="logs-table w-full min-w-[850px] text-[20px] font-bold text-white">
          <thead><tr><th>ID</th><th>Name</th><th>Time</th><th>Date</th><th>Status</th><th>Action</th></tr></thead>
          <tbody>
            {filtered.map((log) => (
              <tr key={log.id}>
                <td>{log.employeeId ?? "—"}</td><td>{log.name}</td><td>{new Date(log.recognizedAt).toLocaleTimeString()}</td><td>{new Date(log.recognizedAt).toLocaleDateString()}</td>
                <td><span className={`status-pill status-${log.status.toLowerCase().replace(" ", "-")}`}>{log.status}</span></td>
                <td>
                  <div className="flex items-center justify-center gap-3">
                    <button
                      aria-label={`View log for ${log.name}`}
                      title="View detail"
                      className="rounded-lg p-1.5 text-[#facc15] transition hover:bg-white/10"
                      onClick={() => setSelected(log)}
                    >
                      <FileSearch size={29} />
                    </button>
                    <button
                      aria-label={`Delete log for ${log.name}`}
                      title="Delete log"
                      className="rounded-lg p-1.5 text-[#ff4d55] transition hover:bg-white/10"
                      onClick={() => { setDeleteError(null); setPendingDelete(log); }}
                    >
                      <Trash2 size={28} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!filtered.length && <p className="py-14 text-center text-lg text-white/70">No matching logs found.</p>}
      </div>
      {selected && <CheckLogDetailModal log={selected} onClose={() => setSelected(null)} />}
      {pendingDelete && (
        <DeleteLogConfirmationModal
          log={pendingDelete}
          deleting={deleting}
          error={deleteError}
          onConfirm={confirmDelete}
          onCancel={() => { setPendingDelete(null); setDeleteError(null); }}
        />
      )}
    </div>
  );
}
