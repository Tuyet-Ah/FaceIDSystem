import { Camera, CameraOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import api from "../services/api";

interface LiveResult {
  status: "SUCCESS" | "UNKNOWN";
  message: string;
  employee?: { name: string } | null;
  similarity?: number;
  threshold?: number;
  metrics?: ModelMetrics;
}

const MODELS = ["buffalo_l", "buffalo_s", "adaface", "ghostfacenet"] as const;
type ModelId = typeof MODELS[number];
interface ModelMetrics {
  model: ModelId;
  pipelineMs: number;
  preparationMs: number;
  queueMs: number;
  coldStart: boolean;
  embeddingsRebuilt: boolean;
  enrolledEmployees: number;
  skippedImages: number;
  threshold: number;
  detector: string;
  detectionScore?: number;
  brightness?: number;
  blurScore?: number;
}
interface Sample { pipeline: number; total: number; similarity?: number; matched: boolean }

export default function LiveAttendanceCamera() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Retain across stop/start: cancelling HTTP cannot cancel server inference.
  const pending = useRef(false);
  const targetFps = useRef(2);
  const lastCompletion = useRef(0);
  const selectedModel = useRef<ModelId>("adaface");
  const recordAttendance = useRef(false);
  const [model, setModel] = useState<ModelId>("adaface");
  const [record, setRecord] = useState(false);
  const [prepared, setPrepared] = useState<ModelMetrics | null>(null);
  const [samples, setSamples] = useState<Partial<Record<ModelId, Sample[]>>>({});
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [fps, setFps] = useState(2);
  const [videoRate, setVideoRate] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<{ latency: number; rate: number } | null>(null);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [message, setMessage] = useState("Bật camera để thử model. Mặc định không ghi điểm danh; mỗi lần chỉ đứng một người trong khung hình.");

  function release() {
    generation.current += 1;
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    stream.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }

  function stop(detail = "Đã tắt camera. Yêu cầu đã gửi trước đó có thể vẫn hoàn tất trên backend.") {
    release();
    setRunning(false);
    setStarting(false);
    setMetrics(null);
    setVideoRate(null);
    setResult(null);
    setMessage(detail);
  }

  useEffect(() => {
    const hide = () => { if (document.hidden) stop("Đã tắt camera khi chuyển tab. Bấm Bật camera để tiếp tục."); };
    document.addEventListener("visibilitychange", hide);
    return () => { document.removeEventListener("visibilitychange", hide); release(); };
  }, []);

  async function scan(id: number) {
    if (id !== generation.current || !stream.current) return;
    const started = performance.now();
    try {
      const element = video.current;
      if (pending.current || !element || element.readyState < 2 || !element.videoWidth) return;
      pending.current = true;
      try {
        const canvas = document.createElement("canvas");
        const scale = Math.min(1, 640 / element.videoWidth);
        canvas.width = Math.round(element.videoWidth * scale);
        canvas.height = Math.round(element.videoHeight * scale);
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Cannot capture camera");
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
        if (id !== generation.current) return;
        if (!blob) throw new Error("Cannot encode frame");
        const form = new FormData();
        form.append("image", blob, "live-frame.jpg");
        form.append("device", "Browser Live Camera");
        form.append("live", "1");
        form.append("model", selectedModel.current);
        form.append("compare", recordAttendance.current ? "0" : "1");
        const response = await api.post<LiveResult>("/recognitions", form, {
          validateStatus: (status) => status === 200 || status === 422,
        });
        if (id !== generation.current) return;
        const completed = performance.now();
        const elapsed = lastCompletion.current ? completed - lastCompletion.current : 0;
        lastCompletion.current = completed;
        setMetrics({ latency: completed - started, rate: elapsed ? 1000 / elapsed : 0 });
        setResult(response.data);
        setMessage(response.data.message);
        const info = response.data.metrics;
        if (info && !info.coldStart && !info.embeddingsRebuilt) {
          setSamples((current) => ({ ...current, [info.model]: [...(current[info.model] ?? []), {
            pipeline: info.pipelineMs, total: completed - started,
            similarity: response.data.similarity, matched: response.data.status === "SUCCESS",
          }].slice(-200) }));
        }
      } finally { pending.current = false; }
    } catch {
      if (id === generation.current) stop("Không thể xử lý ảnh hoặc kết nối backend. Camera đã dừng; kiểm tra backend rồi bật lại.");
    } finally {
      if (id === generation.current && stream.current) {
        timer.current = setTimeout(() => void scan(id), Math.max(0, 1000 / targetFps.current - (performance.now() - started)));
      }
    }
  }

  async function start() {
    release();
    const id = generation.current;
    setStarting(true);
    setResult(null);
    setMetrics(null);
    lastCompletion.current = 0;
    setMessage(`Đang chuẩn bị ${selectedModel.current}: có thể cần tải model và tạo embeddings lần đầu…`);
    try {
      const preparation = await api.post<ModelMetrics>(`/models/${selectedModel.current}/prepare`, undefined, { timeout: 600_000 });
      if (id !== generation.current) return;
      setPrepared(preparation.data);
      setMessage("Model sẵn sàng. Đang chờ quyền truy cập camera…");
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error("Hãy mở bằng Chrome/Edge tại localhost hoặc HTTPS để dùng camera.");
      }
      const media = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }),
          width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 },
        },
      });
      if (id !== generation.current) { media.getTracks().forEach((track) => track.stop()); return; }
      stream.current = media;
      const element = video.current;
      if (!element) { release(); return; }
      element.srcObject = media;
      await element.play();
      if (id !== generation.current) return;
      const track = media.getVideoTracks()[0];
      track.onended = () => stop("Camera đã ngắt kết nối. Kiểm tra thiết bị rồi bật lại.");
      setVideoRate(track.getSettings().frameRate ?? null);
      setDeviceId(track.getSettings().deviceId ?? "");
      setRunning(true);
      setStarting(false);
      setMessage("Đang nhận diện tự động…");
      void scan(id);
      navigator.mediaDevices.enumerateDevices().then((items) => {
        if (id === generation.current) setDevices(items.filter((item) => item.kind === "videoinput"));
      }).catch(() => {});
    } catch (error) {
      if (id !== generation.current) return;
      const name = error instanceof DOMException ? error.name : "";
      const detail = name === "NotAllowedError" ? "Bạn chưa cấp quyền camera. Cho phép camera trên thanh địa chỉ rồi thử lại."
        : name === "NotReadableError" ? "Camera đang bận hoặc không đọc được. Đóng ứng dụng camera khác rồi thử lại."
        : name === "NotFoundError" || name === "OverconstrainedError" ? "Không tìm thấy camera phù hợp. Kiểm tra kết nối hoặc chọn camera mặc định."
        : error instanceof Error ? error.message : "Không mở được camera.";
      stop(detail);
    }
  }

  return (
    <section className="mb-6 rounded-xl bg-white p-5">
      <h3 className="mb-4 flex items-center gap-2 text-xl font-bold text-[#174b6e]"><Camera /> Camera trực tiếp trên web</h3>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(240px,1fr)]">
        <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-950">
          <video ref={video} autoPlay muted playsInline className="h-full w-full object-contain -scale-x-100" />
          {!running && <div className="pointer-events-none absolute inset-0 grid place-items-center text-slate-300"><CameraOff size={56} /></div>}
        </div>
        <div className="space-y-4">
          <label className="block text-sm font-semibold">Model nhận diện
            <select aria-label="Model nhận diện" className="mt-1 w-full rounded border p-2" value={model} onChange={(event) => {
              stop("Đã đổi model. Bấm Bật camera để bắt đầu lượt đo mới.");
              const value = event.target.value as ModelId;
              selectedModel.current = value;
              setModel(value);
              setPrepared(null);
            }}>
              {MODELS.map((name) => <option key={name} value={name}>{name === "adaface" ? "AdaFace IR-50" : name === "ghostfacenet" ? "GhostFaceNet V1 W1.3 S1" : name}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={record} disabled={running || starting} onChange={(event) => { setRecord(event.target.checked); recordAttendance.current = event.target.checked; }} />
            Ghi điểm danh (tắt khi chỉ so sánh)
          </label>
          <div className="flex flex-wrap gap-2">
            {!running && !starting ? <button className="action-button" onClick={() => void start()}>Bật camera</button>
              : <button className="action-button bg-red-700" onClick={() => stop()}>{starting ? "Hủy mở camera" : "Tắt camera"}</button>}
          </div>
          <label className="block text-sm font-semibold">Thiết bị
            <select className="mt-1 w-full rounded border p-2" value={deviceId} disabled={running || starting} onChange={(event) => setDeviceId(event.target.value)}>
              <option value="">Camera mặc định</option>
              {devices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>)}
            </select>
          </label>
          <label className="block text-sm font-semibold">Giới hạn ảnh nhận diện / giây
            <select className="ml-2 rounded border p-2" value={fps} onChange={(event) => { const value = Number(event.target.value); setFps(value); targetFps.current = value; }}>
              {[2, 3, 5].map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
          <p className="text-sm text-slate-600">Video: {videoRate ? `${videoRate.toFixed(0)} FPS (thiết bị báo)` : "mục tiêu 30 FPS"}. Mỗi lần chỉ gửi một ảnh; backend chậm thì tốc độ nhận diện tự giảm.</p>
          {metrics && <p className="text-sm font-semibold">Nhận diện: {metrics.rate.toFixed(1)} ảnh/s · Trễ toàn lượt: {metrics.latency.toFixed(0)} ms</p>}
          {prepared && <p className="text-xs text-slate-600">CPU · Detector: {prepared.detector} · {prepared.enrolledEmployees} nhân viên có embeddings · {prepared.skippedImages} ảnh bị bỏ qua · Chuẩn bị: {prepared.preparationMs.toFixed(0)} ms</p>}
          {result?.metrics && <div className="text-sm text-slate-700">
            <p>Model trả kết quả: <strong>{result.metrics.model}</strong></p>
            <p>Pipeline: {result.metrics.pipelineMs.toFixed(0)} ms · Chờ xử lý: {result.metrics.queueMs.toFixed(0)} ms</p>
            <p>Cosine: {result.similarity?.toFixed(3) ?? "—"} · Ngưỡng: {(result.threshold ?? result.metrics.threshold).toFixed(2)}</p>
            <p>Điểm phát hiện: {result.metrics.detectionScore?.toFixed(3) ?? "—"} · Độ sáng: {result.metrics.brightness ?? "—"} · Độ nét: {result.metrics.blurScore ?? "—"}</p>
          </div>}
          <div role="status" className={`rounded-lg p-3 ${result?.status === "SUCCESS" ? "bg-emerald-50 text-emerald-800" : "bg-slate-100 text-slate-700"}`}>
            {result?.employee && <p className="font-bold">{result.employee.name}</p>}
            <p>{message}</p>
          </div>
          <p className="text-xs text-slate-500">Ảnh được gửi tới backend để nhận diện; chỉ ghi điểm danh khi bật tùy chọn trên. Tắt camera hoặc rời trang để dừng. Chuyển tab cũng tự tắt camera. Đây chưa phải kiểm tra chống giả mạo.</p>
        </div>
      </div>
      <div className="mt-6 overflow-x-auto">
        <div className="mb-3 flex items-center justify-between gap-3"><h4 className="font-bold">So sánh {MODELS.length} model — tối đa 200 lượt gần nhất/model trong trang này</h4><button className="rounded border px-3 py-1 text-sm" onClick={() => setSamples({})}>Xóa thống kê</button></div>
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-100"><tr>{["Model", "Số lượt", "Pipeline TB (ms)", "Pipeline p95 (ms)", "Toàn lượt TB (ms)", "Khớp danh sách", "Cosine cuối"].map((heading) => <th key={heading} className="whitespace-nowrap p-2">{heading}</th>)}</tr></thead>
          <tbody>{MODELS.map((name) => {
            const rows = samples[name] ?? [];
            const ordered = rows.map((row) => row.pipeline).sort((a, b) => a - b);
            const average = (key: "pipeline" | "total") => rows.length ? (rows.reduce((sum, row) => sum + row[key], 0) / rows.length).toFixed(0) : "—";
            return <tr key={name} className="border-b"><td className="p-2 font-semibold">{name}</td><td className="p-2">{rows.length}</td><td className="p-2">{average("pipeline")}</td><td className="p-2">{ordered.length ? ordered[Math.ceil(ordered.length * 0.95) - 1].toFixed(0) : "—"}</td><td className="p-2">{average("total")}</td><td className="p-2">{rows.filter((row) => row.matched).length}/{rows.length}</td><td className="p-2">{rows[rows.length - 1]?.similarity?.toFixed(3) ?? "—"}</td></tr>;
          })}</tbody>
        </table>
        <p className="mt-3 text-xs text-slate-500">Pipeline gồm giải mã ảnh, phát hiện, trích xuất và so khớp; không gồm tải model/tạo embeddings. Số liệu gồm cả khung hình không có mặt. Giữ cùng ánh sáng, khoảng cách, số người và giới hạn ảnh/giây khi so sánh. Đây là so sánh cả pipeline (detector có thể khác), không phải benchmark chỉ backbone. Cosine và số lượt khớp không phải accuracy; cần dữ liệu gán nhãn để đo độ chính xác.</p>
      </div>
    </section>
  );
}
