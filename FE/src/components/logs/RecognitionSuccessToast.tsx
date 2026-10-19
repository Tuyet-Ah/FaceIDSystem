import { CheckCircle2, Clock3, X } from "lucide-react";
import { useEffect } from "react";
import useEmployees from "../../hooks/useEmployees";

export default function RecognitionSuccessToast() {
  const { recognitionNotice, dismissRecognitionNotice } = useEmployees();

  useEffect(() => {
    if (!recognitionNotice) return;
    const timer = window.setTimeout(dismissRecognitionNotice, 6_000);
    return () => window.clearTimeout(timer);
  }, [recognitionNotice, dismissRecognitionNotice]);

  if (!recognitionNotice) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="recognition-toast"
    >
      <div className="h-1.5 bg-emerald-500" />
      <div className="flex gap-4 p-5 pr-12">
        <div className="grid w-12 h-12 shrink-0 self-start place-items-center text-emerald-600">
          <CheckCircle2 size={30} strokeWidth={2.5} />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-extrabold uppercase tracking-wide text-emerald-600">Recognition successful</p>
          <p className="mt-1 truncate text-xl font-extrabold text-[#00233a]">{recognitionNotice.name}</p>
          <p className="text-sm font-semibold text-slate-500">{recognitionNotice.department || "No department"}</p>
          <p className="mt-2 flex items-center gap-1.5 text-sm font-semibold text-slate-600">
            <Clock3 size={15} />
            <span>{new Date(recognitionNotice.recognizedAt).toLocaleString()}</span>
          </p>
          {recognitionNotice.similarity != null && (
            <p className="mt-1 text-sm font-bold text-emerald-700">
              Confidence: {(recognitionNotice.similarity * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>
      <button
        type="button"
        aria-label="Close recognition notification"
        className="absolute right-3 top-4 grid size-8 place-items-center rounded-full text-slate-500 hover:bg-slate-100"
        onClick={dismissRecognitionNotice}
      >
        <X size={20} />
      </button>
    </div>
  );
}
