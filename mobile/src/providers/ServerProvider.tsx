import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { normalizeServer, request } from "../services/client";

interface ServerContextValue {
  server: string;
  ready: boolean;
  storageError: string | null;
  connect: (value: string) => Promise<void>;
}

const ServerContext = createContext<ServerContextValue | null>(null);
const storageKey = "faceid.server";
const defaultServer = "http://10.0.2.2:5000";

export function ServerProvider({ children }: { children: ReactNode }) {
  const [server, setServer] = useState(defaultServer);
  const [ready, setReady] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    AsyncStorage.getItem(storageKey)
      .then((value) => {
        if (active && value) setServer(normalizeServer(value));
      })
      .catch(() => {
        if (active)
          setStorageError("Không đọc được cấu hình đã lưu. Vui lòng kết nối lại máy chủ.");
      })
      .finally(() => {
        if (active) setReady(true);
      });
    return () => {
      active = false;
    };
  }, []);

  async function connect(value: string) {
    const address = normalizeServer(value);
    const health = await request<{ status: string }>(address, "/health");
    if (health.status !== "ok") throw new Error("Máy chủ chưa sẵn sàng");
    await AsyncStorage.setItem(storageKey, address);
    setServer(address);
    setStorageError(null);
  }

  return (
    <ServerContext.Provider value={{ server, ready, storageError, connect }}>
      {children}
    </ServerContext.Provider>
  );
}

export function useServer() {
  const value = useContext(ServerContext);
  if (!value) throw new Error("ServerProvider is required");
  return value;
}
