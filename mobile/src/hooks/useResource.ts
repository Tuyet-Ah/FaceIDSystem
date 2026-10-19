import { useCallback, useEffect, useRef, useState } from "react";
import { AppState } from "react-native";
import { errorMessage } from "../services/client";

export function useResource<T>(load: (signal: AbortSignal) => Promise<T>, refreshInterval = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const controller = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setLoading(true);
    try {
      const result = await load(current.signal);
      if (!current.signal.aborted) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (!current.signal.aborted) setError(errorMessage(e));
    } finally {
      if (!current.signal.aborted) setLoading(false);
      if (controller.current === current) controller.current = null;
    }
  }, [load]);

  useEffect(() => {
    setData(null);
    void refresh();
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active") void refresh();
      else controller.current?.abort();
    });
    const timer =
      refreshInterval > 0
        ? setInterval(() => {
            if (AppState.currentState === "active" && !controller.current) void refresh();
          }, refreshInterval)
        : undefined;
    return () => {
      controller.current?.abort();
      subscription.remove();
      clearInterval(timer);
    };
  }, [refresh, refreshInterval]);

  return { data, error, loading, refresh };
}
