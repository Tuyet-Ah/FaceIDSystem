import { useEffect, useRef, useState } from "react";
import { errorMessage } from "../services/client";

export function useAction() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const locked = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function run(work: () => Promise<void>) {
    if (locked.current) return;
    locked.current = true;
    setPending(true);
    setError(null);
    try {
      await work();
    } catch (e) {
      if (mounted.current) setError(errorMessage(e));
    } finally {
      locked.current = false;
      if (mounted.current) setPending(false);
    }
  }
  return { pending, error, run, clearError: () => setError(null) };
}
