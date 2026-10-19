import { X } from "lucide-react";
import type { ReactNode } from "react";

interface ModalProps {
  children: ReactNode;
  onClose: () => void;
  labelledBy?: string;
  className?: string;
}

export default function Modal({ children, onClose, labelledBy, className = "" }: ModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-[#b4bfce]/85 p-4"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className={`relative max-h-[92vh] overflow-auto rounded-2xl border border-slate-200 bg-white text-[#071f36] shadow-2xl ${className}`}
      >
        <button
          type="button"
          aria-label="Close modal"
          onClick={onClose}
          className="absolute right-5 top-5 z-10 grid size-42 place-items-center rounded-full bg-[#e93f48] text-white shadow-md transition hover:scale-105 sm:right-7 sm:top-6"
        >
          <X size={32} strokeWidth={4} />
        </button>
        {children}
      </section>
    </div>
  );
}
