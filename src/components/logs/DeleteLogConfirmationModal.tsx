import { AlertTriangle } from "lucide-react";
import type { RecognitionLog } from "../../types";

interface Props {
  log: RecognitionLog;
  deleting: boolean;
  error: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteLogConfirmationModal({
  log,
  deleting,
  error,
  onConfirm,
  onCancel,
}: Props) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#001729]/70 p-4">
      <section
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="delete-log-title"
        className="delete-log-modal rounded-xl border border-slate-200 bg-white p-6 text-center text-[#071f36] shadow-2xl"
      >
        <AlertTriangle className="mx-auto mb-2 text-[#e62b35]" size={42} />
        <h2 id="delete-log-title" className="text-[22px] font-extrabold text-[#e62b35]">DELETE SYSTEM LOG</h2>
        <p className="mx-auto mt-3 text-[15px] leading-6 text-slate-700">
          Delete the log for <strong>{log.name}</strong> at{" "}
          <strong>{new Date(log.recognizedAt).toLocaleString()}</strong>?
        </p>
        <p className="mt-1 text-sm text-slate-500">The captured snapshot will also be removed.</p>
        {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</p>}
        <div className="mt-6 flex justify-center gap-4">
          <button
            type="button"
            className="action-button min-w-28 bg-[#e62b35]"
            disabled={deleting}
            onClick={onConfirm}
          >
            {deleting ? "DELETING..." : "DELETE"}
          </button>
          <button type="button" className="action-button min-w-28" disabled={deleting} onClick={onCancel}>CANCEL</button>
        </div>
      </section>
    </div>
  );
}
