const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function loadModule(relativePath, overrides = {}) {
  const filename = path.join(__dirname, "..", relativePath);
  const source = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const exports = {};
  vm.runInNewContext(
    source,
    {
      exports,
      URL,
      Date,
      Error,
      TypeError,
      AbortController,
      setTimeout,
      clearTimeout,
      fetch: globalThis.fetch,
      ...overrides,
    },
    { filename },
  );
  return exports;
}

const validation = loadModule("src/features/employees/validation.ts");

const logFilters = loadModule("src/features/logs/filters.ts");

test("Vietnamese search matches accented names without requiring accents", () => {
  assert.equal(logFilters.normalizeSearch("  Đặng Thị Hồng  "), "dang thi hong");
  const logs = [
    {
      employeeId: 12,
      name: "Đặng Thị Hồng",
      department: "Kế toán",
      status: "SUCCESS",
      recognizedAt: new Date().toISOString(),
    },
  ];
  assert.equal(
    logFilters.filterLogs(logs, {
      query: "dang thi",
      status: "ALL",
      range: "all",
      employeeId: null,
    }).length,
    1,
  );
  assert.equal(
    logFilters.filterLogs(logs, {
      query: "ke toan",
      status: "SUCCESS",
      range: "all",
      employeeId: 12,
    }).length,
    1,
  );
  assert.equal(
    logFilters.filterLogs(logs, { query: "", status: "UNKNOWN", range: "all", employeeId: null })
      .length,
    0,
  );
});

test("history date ranges use local calendar boundaries and exclude tomorrow", () => {
  const now = new Date(2026, 9, 19, 14);
  const timestamp = (day, hour = 0) => new Date(2026, 9, day, hour).toISOString();
  assert.equal(logFilters.matchesTimeRange(timestamp(19), "today", now), true);
  assert.equal(logFilters.matchesTimeRange(timestamp(20), "today", now), false);
  assert.equal(logFilters.matchesTimeRange(timestamp(18, 23), "yesterday", now), true);
  assert.equal(logFilters.matchesTimeRange(timestamp(19), "yesterday", now), false);
  assert.equal(logFilters.matchesTimeRange(timestamp(17), "last3days", now), true);
  assert.equal(logFilters.matchesTimeRange(timestamp(16, 23), "last3days", now), false);
  assert.equal(logFilters.matchesTimeRange(timestamp(13), "last7days", now), true);
});

test("today attendance counts each employee once per shift and only successful logs", () => {
  const now = new Date(2026, 9, 19, 14);
  const log = (
    id,
    employeeId,
    hour,
    attendancePeriod = "MORNING",
    status = "SUCCESS",
    day = 19,
  ) => ({
    id,
    employeeId,
    attendancePeriod,
    status,
    recognizedAt: new Date(2026, 9, day, hour).toISOString(),
  });
  const records = [
    log(1, 1, 8),
    log(2, 1, 9),
    log(3, 1, 13, "AFTERNOON"),
    log(4, 2, 8),
    log(5, 3, 8, "MORNING", "UNKNOWN"),
    log(6, null, 8),
    log(7, 4, 8, "MORNING", "SUCCESS", 18),
  ];
  const result = logFilters.todayCheckIns(records, now);
  assert.equal(result.length, 3);
  assert.equal(result.filter((item) => item.employeeId === 1).length, 2);
  assert.equal(
    result.some((item) => item.id === 1),
    false,
  );
  assert.equal(result.filter((item) => item.attendancePeriod === "MORNING").length, 2);
});

test("image uploads serialize through the installed Expo multipart converter", async () => {
  const bytes = new Uint8Array([255, 216, 255, 217]);
  class LocalFile extends File {
    constructor(uri) {
      super([bytes], uri.split("/").pop(), { type: "image/jpeg" });
    }
  }
  const { appendImage } = loadModule("src/services/faceid.ts", {
    require: (name) => {
      if (name === "expo-file-system") return { File: LocalFile };
      if (name === "react-native") return { Platform: { OS: "ios" } };
      if (name === "./client") return {};
      throw new Error(`Unexpected import ${name}`);
    },
  });
  const { convertFormDataAsync } = loadModule(
    "node_modules/expo/src/winter/fetch/convertFormData.ts",
    {
      require: () => ({ blobToArrayBufferAsync: (blob) => blob.arrayBuffer() }),
      Blob,
      Uint8Array,
      TextEncoder,
    },
  );
  const body = new FormData();
  body.append("name", "Employee");
  appendImage(body, "images", { uri: "file:///cache/front.jpg", mimeType: "image/jpeg" });
  appendImage(body, "images", { uri: "file:///cache/side.jpg", mimeType: "image/jpeg" });
  const serialized = await convertFormDataAsync(body, "test-boundary");
  const text = new TextDecoder().decode(serialized.body);
  assert.match(text, /name="name"/);
  assert.match(text, /name="images"; filename="front.jpg"/);
  assert.match(text, /name="images"; filename="side.jpg"/);
  assert.match(text, /content-type: image\/jpeg/);
  assert.equal((text.match(/name="images"/g) || []).length, 2);
  assert.ok(serialized.body.some((byte) => byte === 255));
});
const fields = {
  name: "Nguyen Van A",
  dateOfBirth: "1998-03-20",
  address: "Ha Noi",
  phone: "0912345678",
  email: "a@example.com",
  department: "IT",
};

test("registration requires 5-10 images; editing can keep existing embeddings", () => {
  assert.equal(Object.keys(validation.validateEmployee(fields, 5, true)).length, 0);
  assert.ok(validation.validateEmployee(fields, 4, true).images);
  assert.ok(validation.validateEmployee(fields, 11, true).images);
  assert.equal(Object.keys(validation.validateEmployee(fields, 0, false)).length, 0);
  assert.ok(validation.validateEmployee(fields, 2, false).images);
});

test("rejects impossible birthdays, future dates and invalid contact data", () => {
  assert.ok(
    validation.validateEmployee({ ...fields, dateOfBirth: "2025-02-29" }, 5, true).dateOfBirth,
  );
  assert.ok(
    validation.validateEmployee({ ...fields, dateOfBirth: "2999-01-01" }, 5, true).dateOfBirth,
  );
  assert.equal(
    validation.validateEmployee({ ...fields, dateOfBirth: "2000-02-29" }, 5, true).dateOfBirth,
    undefined,
  );
  const errors = validation.validateEmployee(
    { ...fields, name: "  ", email: "invalid", phone: "abc" },
    5,
    true,
  );
  assert.ok(errors.name && errors.email && errors.phone);
});

test("server URL normalization rejects credentials and unexpected paths", () => {
  const { normalizeServer } = loadModule("src/services/client.ts");
  assert.equal(normalizeServer(" http://192.168.1.10:5000/api/ "), "http://192.168.1.10:5000");
  for (const value of [
    "invalid",
    "ftp://example.com",
    "http://user:pass@example.com",
    "http://example.com/other",
    "http://example.com/?x=1",
  ]) {
    assert.throws(() => normalizeServer(value));
  }
});

test("API preserves image validation details and handles empty delete responses", async () => {
  const { request, ApiError } = loadModule("src/services/client.ts", {
    fetch: async () =>
      new Response(
        JSON.stringify({
          message: "Invalid images",
          imageErrors: [{ index: 0, fileName: "a.jpg", message: "Blurred" }],
        }),
        { status: 422 },
      ),
  });
  await assert.rejects(request("http://localhost:5000", "/employees"), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 422);
    assert.equal(error.imageErrors[0].message, "Blurred");
    return true;
  });
  const empty = loadModule("src/services/client.ts", {
    fetch: async () => new Response(null, { status: 204 }),
  });
  assert.equal(
    await empty.request("http://localhost:5000", "/employees/1", { method: "DELETE" }),
    undefined,
  );
});

test("API reports non-JSON responses and network failures clearly", async () => {
  const html = loadModule("src/services/client.ts", {
    fetch: async () => new Response("<html>Error</html>", { status: 502 }),
  });
  await assert.rejects(html.request("http://localhost:5000", "/health"), /HTTP 502/);
  const offline = loadModule("src/services/client.ts", {
    fetch: async () => {
      throw new TypeError("fetch failed");
    },
  });
  await assert.rejects(offline.request("http://localhost:5000", "/health"), /Wi-Fi/);
});

test("API propagates cancellation and bounds stalled requests", async () => {
  const pending = loadModule("src/services/client.ts", {
    fetch: (_, { signal }) =>
      new Promise((resolve, reject) => {
        const abort = () => reject(new Error("cancelled"));
        if (signal.aborted) abort();
        else signal.addEventListener("abort", abort);
      }),
  });
  await assert.rejects(
    pending.request("http://localhost:5000", "/health", { timeout: 5 }),
    /quá lâu/,
  );
  const controller = new AbortController();
  const work = pending.request("http://localhost:5000", "/health", { signal: controller.signal });
  controller.abort();
  await assert.rejects(work, /đã được hủy/);
});

test("request finishes even when native response body ignores abort", async () => {
  const { request, ApiError } = loadModule("src/services/client.ts", {
    fetch: async () => ({ status: 422, ok: false, text: () => new Promise(() => {}) }),
    setTimeout: (callback) => setTimeout(callback, 10),
  });
  await assert.rejects(request("http://localhost:5000", "/employees"), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 422);
    return true;
  });
});

test("request timeout also works when transport ignores AbortSignal", async () => {
  const { request } = loadModule("src/services/client.ts", {
    fetch: () => new Promise(() => {}),
  });
  await assert.rejects(request("http://localhost:5000", "/health", { timeout: 5 }), /quá lâu/);
});

test("live capture loop waits for slow requests and cancels pending work", async () => {
  let clock = 0;
  let scheduled;
  let delay;
  let finish;
  let signal;
  let calls = 0;
  const { startSerialLoop } = loadModule("src/utils/serialLoop.ts", {
    Date: { now: () => clock },
    setTimeout: (callback, milliseconds) => {
      scheduled = callback;
      delay = milliseconds;
      return 1;
    },
    clearTimeout: () => {
      scheduled = undefined;
    },
  });
  const stop = startSerialLoop(
    (current) => {
      calls += 1;
      signal = current;
      return new Promise((resolve) => {
        finish = resolve;
      });
    },
    1000,
    assert.fail,
  );
  assert.equal(calls, 1);
  assert.equal(scheduled, undefined);
  clock = 1500;
  finish();
  await new Promise(setImmediate);
  assert.equal(delay, 0);
  scheduled();
  assert.equal(calls, 2);
  stop();
  assert.equal(signal.aborted, true);
  finish();
  await new Promise(setImmediate);
  assert.equal(scheduled, undefined);
});

test("live capture respects one-second cadence and recovers after errors", async () => {
  let clock = 0;
  let scheduled;
  let delay;
  let calls = 0;
  let errors = 0;
  const { startSerialLoop } = loadModule("src/utils/serialLoop.ts", {
    Date: { now: () => clock },
    setTimeout: (callback, milliseconds) => {
      scheduled = callback;
      delay = milliseconds;
      return 1;
    },
    clearTimeout: () => {
      scheduled = undefined;
    },
  });
  const stop = startSerialLoop(
    async () => {
      calls += 1;
      clock += 200;
      if (calls === 1) throw new Error("offline");
    },
    1000,
    () => {
      errors += 1;
    },
  );
  await new Promise(setImmediate);
  assert.equal(errors, 1);
  assert.equal(delay, 800);
  scheduled();
  await new Promise(setImmediate);
  assert.equal(calls, 2);
  assert.equal(delay, 800);
  stop();
  assert.equal(scheduled, undefined);
});
