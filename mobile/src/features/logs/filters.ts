import type { RecognitionLog } from "../../types";

export type TimeRange = "all" | "today" | "yesterday" | "last3days" | "last7days";

export function normalizeSearch(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[đĐ]/g, "d")
    .toLowerCase()
    .trim();
}

export function matchesTimeRange(value: string, range: TimeRange, now = new Date()) {
  if (range === "all") return true;
  const moment = new Date(value);
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  if (range === "yesterday") {
    end.setTime(start.getTime());
    start.setDate(start.getDate() - 1);
  }
  if (range === "last3days") start.setDate(start.getDate() - 2);
  if (range === "last7days") start.setDate(start.getDate() - 6);
  return moment >= start && moment < end;
}

export function filterLogs(
  logs: RecognitionLog[],
  options: { query: string; status: string; range: TimeRange; employeeId: number | null },
  now = new Date(),
) {
  const query = normalizeSearch(options.query);
  return logs.filter(
    (log) =>
      (options.employeeId === null || log.employeeId === options.employeeId) &&
      (options.status === "ALL" || log.status === options.status) &&
      matchesTimeRange(log.recognizedAt, options.range, now) &&
      (!query ||
        normalizeSearch(`${log.employeeId ?? ""} ${log.name} ${log.department}`).includes(query)),
  );
}

export function todayCheckIns(logs: RecognitionLog[], now = new Date()) {
  const unique = new Map<string, RecognitionLog>();
  const ordered = [...logs].sort((a, b) => Date.parse(b.recognizedAt) - Date.parse(a.recognizedAt));
  for (const log of ordered) {
    if (
      log.status !== "SUCCESS" ||
      log.employeeId === null ||
      !matchesTimeRange(log.recognizedAt, "today", now)
    )
      continue;
    const period =
      log.attendancePeriod ??
      (new Date(log.recognizedAt).getHours() < 12 ? "MORNING" : "AFTERNOON");
    const key = `${log.employeeId}:${period}`;
    if (!unique.has(key)) unique.set(key, { ...log, attendancePeriod: period });
  }
  return [...unique.values()];
}
