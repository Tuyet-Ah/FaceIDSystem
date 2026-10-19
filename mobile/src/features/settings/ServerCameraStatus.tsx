import { useCallback } from "react";
import { Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useServer } from "../../providers/ServerProvider";
import { useResource } from "../../hooks/useResource";
import { request } from "../../services/client";
import type { CameraStatus } from "../../types";
import { ErrorNotice, LoadingState } from "../../components/Feedback";
import { colors, layout } from "../../theme";

const states: Record<CameraStatus["state"], string> = {
  STARTING: "Đang khởi động",
  RUNNING: "Đang hoạt động",
  STOPPED: "Đã dừng",
  ERROR: "Lỗi camera",
};

export default function ServerCameraStatus() {
  const { server } = useServer();
  const status = useResource(
    useCallback(
      (signal: AbortSignal) => request<CameraStatus>(server, "/camera/status", { signal }),
      [server],
    ),
    5_000,
  );
  return (
    <View style={{ gap: 12 }}>
      <View style={layout.row}>
        <Ionicons name="desktop-outline" size={24} color={colors.accent} />
        <Text style={layout.subtitle}>Camera máy chủ</Text>
      </View>
      <ErrorNotice message={status.error} retry={() => void status.refresh()} />
      {!status.data && status.loading && <LoadingState />}
      {status.data && (
        <>
          <Text style={{ color: status.data.running ? colors.primary : colors.muted }}>
            {states[status.data.state]}
          </Text>
          <ErrorNotice message={status.data.error} />
          {status.data.lastRecognition && (
            <View style={{ gap: 4 }}>
              <Text style={layout.text}>{status.data.lastRecognition.employee.name}</Text>
              <Text style={layout.muted}>
                {new Date(status.data.lastRecognition.recognizedAt).toLocaleString("vi-VN")}
              </Text>
            </View>
          )}
        </>
      )}
    </View>
  );
}
