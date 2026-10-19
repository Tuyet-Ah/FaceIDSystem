// Keep capture starts at least intervalMs apart, without queuing slow requests.
export function startSerialLoop(
  task: (signal: AbortSignal) => Promise<void>,
  intervalMs: number,
  onError: (error: unknown) => void,
) {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;

  async function tick() {
    if (controller.signal.aborted) return;
    const started = Date.now();
    try {
      await task(controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) onError(error);
    } finally {
      if (!controller.signal.aborted) {
        timer = setTimeout(() => void tick(), Math.max(0, intervalMs - (Date.now() - started)));
      }
    }
  }

  void tick();
  return () => {
    controller.abort();
    clearTimeout(timer);
  };
}
