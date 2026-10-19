import { ImagePlus, Star, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import type { Employee, EmployeeInput } from "../../types";
import { mediaUrl } from "../../services/api";

interface Props {
  employee?: Employee;
  onCancel: () => void;
  onSave: (value: EmployeeInput) => Promise<void>;
}

type FormFields = Omit<EmployeeInput, "images" | "primaryImageIndex">;

const emptyForm: FormFields = {
  name: "", dateOfBirth: "", address: "", phone: "", email: "", department: "",
};

const fields = [
  ["name", "Name", "Enter employee’s full name", "text"],
  ["dateOfBirth", "Date of birth", "", "date"],
  ["address", "Address", "Enter employee’s address", "text"],
  ["phone", "Phone", "Enter employee’s phone number", "tel"],
  ["email", "Email", "Enter employee’s email", "email"],
  ["department", "Department", "Enter employee’s department", "text"],
] as const;

export default function EmployeeForm({ employee, onCancel, onSave }: Props) {
  const initialFields: FormFields = employee ? {
    name: employee.name,
    dateOfBirth: employee.dateOfBirth,
    address: employee.address,
    phone: employee.phone,
    email: employee.email,
    department: employee.department,
  } : emptyForm;
  const [form, setForm] = useState<FormFields>(initialFields);
  const [images, setImages] = useState<File[]>([]);
  const [primaryImageIndex, setPrimaryImageIndex] = useState(0);
  const [errors, setErrors] = useState<Partial<Record<keyof FormFields | "images" | "submit", string>>>({});
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previews = useMemo(() => images.map((file) => URL.createObjectURL(file)), [images]);

  useEffect(() => () => previews.forEach(URL.revokeObjectURL), [previews]);

  const update = (event: ChangeEvent<HTMLInputElement>) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
    setErrors((current) => ({ ...current, [event.target.name]: undefined }));
  };

  const upload = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    const accepted = selected.filter((file) => file.type.startsWith("image/") && file.size <= 10 * 1024 * 1024);
    const merged = [...images, ...accepted];
    const unique = merged.filter((file, index, all) =>
      all.findIndex((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified) === index,
    );
    if (unique.length > 10) {
      setErrors((current) => ({ ...current, images: "Maximum 10 images" }));
      event.target.value = "";
      return;
    }
    setImages(unique);
    if (!images.length && unique.length) setPrimaryImageIndex(0);
    setErrors((current) => ({ ...current, images: undefined, submit: undefined }));
    event.target.value = "";
  };

  const removeImage = (index: number) => {
    setImages((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setPrimaryImageIndex((current) => {
      if (current === index) return 0;
      return current > index ? current - 1 : current;
    });
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const nextErrors: typeof errors = {};
    fields.forEach(([key]) => {
      if (!form[key].trim()) nextErrors[key] = "Required";
    });
    if (!employee && images.length < 5) nextErrors.images = "Please select at least 5 face images";
    if (employee && images.length > 0 && images.length < 5) nextErrors.images = "Select at least 5 images when retraining";
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors);
      return;
    }
    setSaving(true);
    try {
      await onSave({
        ...form,
        images,
        primaryImageIndex: images.length ? primaryImageIndex : null,
      });
    } catch (error: any) {
      const body = error?.response?.data;
      const details = body?.imageErrors?.map((item: { fileName: string; message: string }) => `${item.fileName}: ${item.message}`).join("; ");
      setErrors((current) => ({ ...current, submit: details || body?.message || "Unable to save employee" }));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit} className="grid min-h-[518px] grid-cols-[minmax(460px,1fr)_minmax(360px,480px)] gap-10 rounded-[11px] bg-white px-[52px] py-[28px]">
      <div>
        {employee && (
          <div className="form-row">
            <label>ID</label>
            <span className="px-3 text-[20px] font-bold">{employee.id}</span>
          </div>
        )}
        {fields.map(([key, label, placeholder, type]) => (
          <div className="form-row" key={key}>
            <label htmlFor={key}>{label}</label>
            <div className="w-full">
              <input
                id={key}
                name={key}
                type={type}
                value={form[key]}
                onChange={update}
                placeholder={placeholder}
                aria-invalid={Boolean(errors[key])}
                className={errors[key] ? "border-red-500" : ""}
              />
              {errors[key] && <p className="mt-1 text-sm text-red-600">{errors[key]}</p>}
            </div>
          </div>
        ))}
      </div>
      <div className="flex flex-col gap-4 py-2">
        <input ref={inputRef} className="hidden" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={upload} />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className={`flex min-h-24 items-center justify-center gap-3 rounded-xl border-2 border-dashed px-5 text-lg font-bold text-[#168fd7] ${errors.images ? "border-red-500" : "border-[#4d69c4]"}`}
        >
          <Upload size={26} /> Add face images (5–10 total)
        </button>
        <div className="flex items-center justify-between text-sm font-bold">
          <span className={images.length >= 5 ? "text-emerald-600" : "text-slate-600"}>
            {images.length}/5 required images
          </span>
          {employee && !images.length && <span className="text-slate-500">Keeping {employee.faceImageCount} trained images</span>}
        </div>
        <p className="text-sm text-slate-600">Use clear photos: front, left, right and slightly different angles. Select the star on one photo to use it as the profile photo.</p>
        <div className="grid max-h-[270px] grid-cols-3 gap-2 overflow-y-auto">
          {previews.map((preview, index) => (
            <div
              key={`${images[index].name}-${images[index].size}`}
              className={`group relative aspect-square overflow-hidden rounded-lg bg-slate-100 ${primaryImageIndex === index ? "ring-4 ring-amber-400 ring-inset" : ""}`}
            >
              <img src={preview} alt={`Face ${index + 1}`} className="size-full object-cover" />
              <button
                type="button"
                aria-label={`Use image ${index + 1} as profile photo`}
                aria-pressed={primaryImageIndex === index}
                title="Set as profile photo"
                className={`absolute left-1 top-1 grid size-8 place-items-center rounded-full shadow-md ${primaryImageIndex === index ? "bg-amber-400 text-white" : "bg-white/90 text-slate-500 hover:text-amber-500"}`}
                onClick={() => setPrimaryImageIndex(index)}
              >
                <Star size={18} fill={primaryImageIndex === index ? "currentColor" : "none"} />
              </button>
              <button
                type="button"
                aria-label={`Remove image ${index + 1}`}
                className="absolute right-1 top-1 rounded-full bg-red-600 p-1 text-white"
                onClick={() => removeImage(index)}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
          {!previews.length && employee?.primaryPhotoUrl && (
            <div className="relative aspect-square overflow-hidden rounded-lg ring-4 ring-amber-400 ring-inset">
              <img src={mediaUrl(employee.primaryPhotoUrl)} alt={employee.name} className="size-full object-cover" />
              <span className="absolute bottom-2 left-2 rounded-full bg-amber-400 px-2.5 py-1 text-xs font-bold text-white shadow">Current profile photo</span>
            </div>
          )}
          {!previews.length && !employee && (
            <div className="col-span-3 grid h-32 place-items-center rounded-lg bg-slate-100 text-slate-400"><ImagePlus size={45} /></div>
          )}
        </div>
        {errors.images && <p className="text-sm font-semibold text-red-600">{errors.images}</p>}
        {errors.submit && <p className="rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">{errors.submit}</p>}
        <div className="mt-auto flex justify-end gap-5">
          <button type="button" className="action-button" onClick={onCancel}>CANCEL</button>
          <button type="submit" className="action-button" disabled={saving}>{saving ? "TRAINING..." : "SAVE"}</button>
        </div>
      </div>
    </form>
  );
}
