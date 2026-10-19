import { Camera } from "lucide-react";
import { mediaUrl } from "../../services/api";
import type { RecognitionLog } from "../../types";
import Modal from "../common/Modal";

interface Props {
  log: RecognitionLog;
  onClose: () => void;
}

export default function CheckLogDetailModal({ log, onClose }: Props) {
  const confidence = log.similarity == null ? "N/A" : `${(log.similarity * 100).toFixed(1)}%`;
  const recognizedAt = new Date(log.recognizedAt);
  const isSuccess = log.status === "SUCCESS";

  const details = isSuccess
    ? [
        { label: "Employee", value: log.name || "Unknown" },
        { label: "Employee ID", value: log.employeeId ?? "Unknown" },
        { label: "Department", value: log.department || "Not specified" },
        { label: "Check-in", value: recognizedAt.toLocaleTimeString() },
        { label: "Date", value: recognizedAt.toLocaleDateString() },
        { label: "Confidence", value: confidence },
        { label: "Device", value: log.device || "Unknown device" },
      ]
    : [
        { label: "Person", value: "Unknown" },
        { label: "First seen", value: new Date(log.firstSeenAt || log.recognizedAt).toLocaleString() },
        { label: "Last seen", value: new Date(log.lastSeenAt || log.recognizedAt).toLocaleString() },
        { label: "Observations", value: log.occurrenceCount || 1 },
        { label: "Similarity", value: confidence },
        { label: "Device", value: log.device || "Unknown device" },
      ];

  return (
    <Modal
      onClose={onClose}
      labelledBy="log-title"
      className="recognition-detail-modal w-full max-w-[980px]"
    >
      <div className="recognition-detail-heading">
        <p>System log</p>
        <h2 id="log-title">Recognition detail</h2>
      </div>

      <div className="recognition-detail-content">
        <section className="recognition-detail-info" aria-label="Recognition information">
          <dl>
            {details.map(({ label, value }) => (
              <div className="recognition-detail-row" key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
            <div className="recognition-detail-row">
              <dt>Status</dt>
              <dd>
                <span className={`recognition-status ${isSuccess ? "is-success" : "is-unknown"}`}>
                  {isSuccess ? "SUCCESS" : "UNKNOWN"}
                </span>
              </dd>
            </div>
          </dl>

          {log.reason && (
            <div className="recognition-reason">
              <span>Reason</span>
              <p>{log.reason}</p>
            </div>
          )}
        </section>

        <figure className="recognition-snapshot">
          <div className="recognition-snapshot-frame">
            {log.imageUrl ? (
              <img src={mediaUrl(log.imageUrl)} alt={`Recognition snapshot of ${log.name}`} />
            ) : (
              <div className="recognition-snapshot-empty">
                <Camera size={46} />
                <span>No snapshot available</span>
              </div>
            )}
          </div>
          <figcaption>
            <strong>{log.name || "Unknown"}</strong>
            <span>Captured by {log.device || "recognition device"}</span>
          </figcaption>
        </figure>
      </div>
    </Modal>
  );
}
