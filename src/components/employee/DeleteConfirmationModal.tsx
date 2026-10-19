import { AlertTriangle } from "lucide-react";

interface Props {
  employeeName: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteConfirmationModal({ employeeName, onConfirm, onCancel }: Props) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#001729]/70 p-4">
      <section role="alertdialog" aria-modal="true" className="w-full max-w-[700px] rounded-xl bg-white p-10 text-center shadow-2xl">
        <AlertTriangle className="mx-auto mb-3 text-[#f44336]" size={54} />
        <h2 className="text-[28px] font-bold text-[#ee1f29]">WARNING</h2>
        <p className="mx-auto mt-4 max-w-[580px] text-[20px] leading-7 text-[#1a2d42]">
          Are you sure you want to delete <strong>{employeeName}</strong>? If you do,
          their face images and embeddings will be removed. Existing recognition logs are retained.
        </p>
        <div className="mt-8 flex justify-center gap-8">
          <button type="button" className="action-button min-w-32 bg-[#e62b35]" onClick={onConfirm}>YES</button>
          <button type="button" className="action-button min-w-32" onClick={onCancel}>NO</button>
        </div>
      </section>
    </div>
  );
}
