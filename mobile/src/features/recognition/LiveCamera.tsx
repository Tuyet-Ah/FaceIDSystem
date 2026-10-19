import { useEffect, useRef, useState } from "react";
import { AppState, Linking, Pressable, StyleSheet, Text, View } from "react-native";
import { CameraView, useCameraPermissions, type CameraType } from "expo-camera";
import { File } from "expo-file-system";
import { Ionicons } from "@expo/vector-icons";
import { useServer } from "../../providers/ServerProvider";
import { recognize } from "../../services/faceid";
import { errorMessage } from "../../services/client";
import { startSerialLoop } from "../../utils/serialLoop";
import type { RecognitionResult } from "../../types";
import Button from "../../components/Button";
import { ErrorNotice, LoadingState } from "../../components/Feedback";
import RecognitionResultView from "./RecognitionResultView";
import { colors, layout } from "../../theme";

export default function LiveCamera() {
  const { server } = useServer();
  const [permission, requestPermission, refreshPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const inFlight = useRef(false);
  const [active, setActive] = useState(AppState.currentState === "active");
  const [ready, setReady] = useState(false);
  const [facing, setFacing] = useState<CameraType>("front");
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [lastTime, setLastTime] = useState<string | null>(null);

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (state) => {
      setActive(state === "active");
      if (state === "active") {
        void refreshPermission().catch((failure) => setError(errorMessage(failure)));
      }
      if (state !== "active") {
        setRunning(false);
        setReady(false);
      }
    });
    return () => subscription.remove();
  }, [refreshPermission]);

  useEffect(() => {
    if (!running || !ready || !active || !permission?.granted) return;
    return startSerialLoop(
      async (signal) => {
        if (inFlight.current || !camera.current) return;
        inFlight.current = true;
        setBusy(true);
        let uri: string | undefined;
        try {
          const photo = await camera.current.takePictureAsync({
            quality: 0.5,
            shutterSound: false,
          });
          uri = photo?.uri;
          if (signal.aborted) return;
          if (!uri) throw new Error("Không chụp được ảnh từ camera");
          const recognition = await recognize(
            server,
            { uri, mimeType: "image/jpeg" },
            { signal, live: true },
          );
          if (!signal.aborted) {
            setResult(recognition);
            setLastTime(new Date().toLocaleTimeString("vi-VN"));
            setError(null);
          }
        } finally {
          if (uri) {
            try {
              const file = new File(uri);
              if (file.exists) file.delete();
            } catch {
              /* Cache cleanup must not interrupt recognition. */
            }
          }
          inFlight.current = false;
          if (!signal.aborted) setBusy(false);
        }
      },
      1000,
      (failure) => {
        setError(errorMessage(failure));
        setResult(null);
        setBusy(false);
      },
    );
  }, [server, running, ready, active, permission?.granted]);

  if (!permission) return <LoadingState />;
  if (!permission.granted)
    return (
      <View style={{ gap: 16 }}>
        <Text style={layout.text}>Cần quyền camera để nhận diện trực tiếp.</Text>
        <Button
          title={permission.canAskAgain ? "Cho phép camera" : "Mở cài đặt"}
          icon="camera-outline"
          onPress={() => {
            if (permission.canAskAgain)
              void requestPermission().catch((failure) => setError(errorMessage(failure)));
            else void Linking.openSettings().catch((failure) => setError(errorMessage(failure)));
          }}
        />
        <ErrorNotice message={error} />
      </View>
    );

  return (
    <View style={{ gap: 16 }}>
      <View style={styles.preview}>
        {active && (
          <CameraView
            key={facing}
            ref={camera}
            style={StyleSheet.absoluteFill}
            facing={facing}
            mode="picture"
            onCameraReady={() => {
              setReady(true);
              setError(null);
            }}
            onMountError={(failure) => {
              setReady(false);
              setRunning(false);
              setError(failure.message);
            }}
          />
        )}
        <View pointerEvents="none" style={styles.status}>
          <View style={[styles.dot, { backgroundColor: running ? "#5ae0b8" : "#ffffff" }]} />
          <Text style={styles.statusText}>
            {running ? (busy ? "Đang xác nhận..." : "Đang nhận diện") : "Đã tạm dừng"}
          </Text>
        </View>
      </View>
      <ErrorNotice message={error} />
      <View style={layout.row}>
        <View style={layout.fill}>
          <Button
            title={running ? "Dừng" : "Bắt đầu"}
            icon={running ? "stop-circle-outline" : "play-outline"}
            disabled={!active || !ready}
            onPress={() => {
              setBusy(false);
              setRunning((current) => !current);
            }}
          />
        </View>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Đổi camera trước/sau"
          disabled={running}
          style={[styles.flip, running && { opacity: 0.4 }]}
          onPress={() => {
            setReady(false);
            setResult(null);
            setLastTime(null);
            setFacing((current) => (current === "front" ? "back" : "front"));
          }}
        >
          <Ionicons name="camera-reverse-outline" size={26} color={colors.primary} />
        </Pressable>
      </View>
      {!!lastTime && <Text style={layout.muted}>Cập nhật: {lastTime}</Text>}
      <RecognitionResultView result={result} />
    </View>
  );
}

const styles = StyleSheet.create({
  preview: {
    width: "100%",
    aspectRatio: 3 / 4,
    maxHeight: 480,
    backgroundColor: "#152429",
    borderRadius: 6,
    overflow: "hidden",
  },
  status: {
    position: "absolute",
    top: 12,
    left: 12,
    right: 12,
    backgroundColor: "rgba(0,0,0,0.65)",
    padding: 10,
    borderRadius: 4,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: "white", fontSize: 13, flexShrink: 1 },
  flip: {
    minWidth: 48,
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 6,
    backgroundColor: colors.primarySoft,
  },
});
