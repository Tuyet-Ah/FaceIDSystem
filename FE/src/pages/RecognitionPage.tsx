import { Camera, CameraOff, ImageUp, ScanFace } from "lucide-react";
import { useEffect, useRef, useState, type ChangeEvent } from "react";
import useEmployees from "../hooks/useEmployees";
import { mediaUrl } from "../services/api";
import { recognitionService } from "../services/recognitionService";
import type { RecognitionResult } from "../types";

function cameraErrorMessage(error: unknown, embedded: boolean) {
  if (!window.isSecureContext) {
    return "Camera requires a secure page. Open this app at http://localhost:8443 or http://127.0.0.1:8443, not through a LAN IP over HTTP.";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser context cannot access the camera. Open the app directly in Chrome/Edge instead of an embedded preview.";
  }

  const name = error instanceof DOMException ? error.name : "UnknownError";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return embedded
      ? "The embedded preview does not allow camera access. Use Open camera tab below and grant permission there."
      : "The browser denied camera access for this site. Check the camera icon in the address bar and Windows camera privacy settings.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No available camera was found. Close other camera apps, reconnect the camera, then try again.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is being used by another application. Close Camera, Teams, Zoom or another browser tab and retry.";
  }
  if (name === "OverconstrainedError") {
    return "The selected camera does not support the requested mode. Choose another camera or retry with the default camera.";
  }
  return `Unable to start the camera (${name}). Please retry or upload a photo.`;
}

export default function RecognitionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  const [scanning, setScanning] = useState(false);
  const [message, setMessage] = useState("Start the camera or upload a photo to recognize a face.");
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const { refreshLogs } = useEmployees();

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  };

  useEffect(() => () => stopCamera(), []);

  const startCamera = async (deviceId = selectedCamera) => {
    setCameraError(null);
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const detail = cameraErrorMessage(new DOMException("Unsupported", "SecurityError"), window.self !== window.top);
      setCameraError(detail);
      setMessage(detail);
      return;
    }

    stopCamera();
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (error) {
        const name = error instanceof DOMException ? error.name : "";
        if (name !== "OverconstrainedError" && name !== "NotFoundError") throw error;
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const available = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === "videoinput");
      setCameras(available);
      const activeDevice = stream.getVideoTracks()[0]?.getSettings().deviceId;
      if (activeDevice) setSelectedCamera(activeDevice);
      setCameraOn(true);
      setMessage("Center one face in the frame, then press Scan face.");
    } catch (error) {
      const detail = cameraErrorMessage(error, window.self !== window.top);
      setCameraError(detail);
      setMessage(detail);
    }
  };

  const submitImage = async (file: File, device: string) => {
    setScanning(true);
    setResult(null);
    try {
      const response = await recognitionService.recognize(file, device);
      setResult(response.data);
      setMessage(response.data.message);
      await refreshLogs();
    } catch (error: any) {
      const body = error?.response?.data;
      setMessage(body?.message || "Unable to recognize this image.");
      if (body?.status) setResult(body as RecognitionResult);
      await refreshLogs();
    } finally {
      setScanning(false);
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setMessage("Camera is not ready yet.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) void submitImage(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }), "Browser Camera");
    }, "image/jpeg", 0.92);
  };

  const upload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void submitImage(file, "Uploaded Photo");
    event.target.value = "";
  };

  const success = result?.status === "SUCCESS" && result.employee;

  return (
    <div className="page-shell">
      <h2 className="page-title mb-8 px-4">Face recognition</h2>
      <section className="grid min-h-[570px] gap-8 rounded-[11px] bg-white p-8 lg:grid-cols-[minmax(480px,1.5fr)_minmax(300px,1fr)]">
        <div>
          <div className="relative aspect-video overflow-hidden rounded-xl bg-[#071f36]">
            <video ref={videoRef} autoPlay playsInline muted className="size-full object-cover" />
            {!cameraOn && <div className="absolute inset-0 grid place-items-center text-white/70"><ScanFace size={110} strokeWidth={1.2} /></div>}
          </div>
          <div className="mt-5 flex flex-wrap gap-3">
            {!cameraOn ? (
              <button type="button" className="action-button flex items-center gap-2" onClick={() => void startCamera()}><Camera size={20} /> START CAMERA</button>
            ) : (
              <>
                <button type="button" className="action-button flex items-center gap-2" onClick={capture} disabled={scanning}><ScanFace size={20} /> {scanning ? "SCANNING..." : "SCAN FACE"}</button>
                <button type="button" className="action-button flex items-center gap-2 bg-red-700" onClick={stopCamera}><CameraOff size={20} /> STOP</button>
              </>
            )}
            <label className="action-button flex cursor-pointer items-center gap-2">
              <ImageUp size={20} /> UPLOAD PHOTO
              <input type="file" accept="image/*" className="hidden" onChange={upload} />
            </label>
          </div>
          {cameras.length > 1 && (
            <label className="mt-4 block text-sm font-bold text-slate-700">
              Camera
              <select
                className="ml-3 rounded-lg border border-slate-300 bg-white px-3 py-2"
                value={selectedCamera}
                onChange={(event) => {
                  const deviceId = event.target.value;
                  setSelectedCamera(deviceId);
                  void startCamera(deviceId);
                }}
              >
                {cameras.map((camera, index) => <option key={camera.deviceId} value={camera.deviceId}>{camera.label || `Camera ${index + 1}`}</option>)}
              </select>
            </label>
          )}
          {cameraError && (
            <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm font-semibold text-red-700">
              <p>{cameraError}</p>
              {window.self !== window.top && (
                <button type="button" className="mt-3 rounded-lg bg-[#174b6e] px-4 py-2 text-white" onClick={() => window.open(window.location.href, "_blank", "noopener,noreferrer")}>
                  OPEN CAMERA TAB
                </button>
              )}
            </div>
          )}
          <p className="mt-4 rounded-lg bg-slate-100 p-3 font-semibold text-slate-700">{message}</p>
        </div>

        <div className={`rounded-xl border-2 p-6 ${success ? "border-emerald-500 bg-emerald-50" : result ? "border-amber-500 bg-amber-50" : "border-slate-200 bg-slate-50"}`}>
          <h3 className="mb-6 text-2xl font-bold text-[#174b6e]">Recognition result</h3>
          {success ? (
            <div className="space-y-4">
              <img src={mediaUrl(result.employee!.primaryPhotoUrl)} alt={result.employee!.name} className="mx-auto h-64 w-52 rounded-xl object-cover" />
              <p className="text-center text-2xl font-bold text-emerald-700">{result.employee!.name}</p>
              <dl className="space-y-2 text-lg">
                <div className="detail-row"><dt className="font-bold">ID</dt><dd>{result.employee!.id}</dd></div>
                <div className="detail-row"><dt className="font-bold">Department</dt><dd>{result.employee!.department}</dd></div>
                <div className="detail-row"><dt className="font-bold">Confidence</dt><dd>{((result.similarity ?? 0) * 100).toFixed(1)}%</dd></div>
              </dl>
            </div>
          ) : result ? (
            <div className="grid min-h-72 place-items-center text-center">
              <div><CameraOff className="mx-auto mb-4 text-amber-600" size={70} /><p className="text-2xl font-bold">{result.status}</p><p className="mt-2 text-slate-600">{result.message}</p></div>
            </div>
          ) : (
            <div className="grid min-h-72 place-items-center text-center text-slate-400"><ScanFace size={90} strokeWidth={1.2} /></div>
          )}
        </div>
      </section>
    </div>
  );
}
