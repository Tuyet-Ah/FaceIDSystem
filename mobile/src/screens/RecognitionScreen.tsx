import { useState } from "react";
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type { ImagePickerAsset } from "expo-image-picker";
import type { RecognitionResult } from "../types";
import { useServer } from "../providers/ServerProvider";
import { useAction } from "../hooks/useAction";
import { recognize } from "../services/faceid";
import { pickImages } from "../utils/images";
import Button from "../components/Button";
import { ErrorNotice } from "../components/Feedback";
import LiveCamera from "../features/recognition/LiveCamera";
import RecognitionResultView from "../features/recognition/RecognitionResultView";
import { colors, layout } from "../theme";

export default function RecognitionScreen() {
  const [mode, setMode] = useState<"live" | "photo">("live");
  return (
    <ScrollView contentContainerStyle={layout.content}>
      <Text style={layout.title}>Điểm danh</Text>
      <View style={styles.modes} accessibilityRole="tablist">
        {(
          [
            { key: "live", label: "Trực tiếp", icon: "videocam-outline" },
            { key: "photo", label: "Ảnh", icon: "image-outline" },
          ] as const
        ).map((item) => (
          <Pressable
            key={item.key}
            accessibilityRole="tab"
            accessibilityState={{ selected: mode === item.key }}
            onPress={() => setMode(item.key)}
            style={[styles.mode, mode === item.key && styles.selected]}
          >
            <Ionicons
              name={item.icon}
              size={20}
              color={mode === item.key ? "white" : colors.muted}
            />
            <Text style={{ color: mode === item.key ? "white" : colors.muted }}>{item.label}</Text>
          </Pressable>
        ))}
      </View>
      {mode === "live" ? <LiveCamera /> : <PhotoRecognition />}
    </ScrollView>
  );
}

function PhotoRecognition() {
  const { server } = useServer();
  const [photo, setPhoto] = useState<ImagePickerAsset | null>(null);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const action = useAction();

  function choose(camera: boolean) {
    void action.run(async () => {
      const images = await pickImages(camera);
      if (images) {
        setPhoto(images[0]);
        setResult(null);
      }
    });
  }

  return (
    <View style={{ gap: 16 }}>
      <ErrorNotice message={action.error} />
      <View style={styles.preview}>
        {photo ? (
          <Image source={{ uri: photo.uri }} style={styles.image} resizeMode="contain" />
        ) : (
          <View style={{ alignItems: "center", gap: 12 }}>
            <Ionicons name="scan-outline" size={80} color={colors.primary} />
            <Text style={layout.muted}>Chưa có ảnh</Text>
          </View>
        )}
      </View>
      <View style={layout.actions}>
        <Button
          title="Chụp ảnh"
          icon="camera-outline"
          disabled={action.pending}
          onPress={() => choose(true)}
        />
        <Button
          title="Thư viện"
          icon="images-outline"
          variant="secondary"
          disabled={action.pending}
          onPress={() => choose(false)}
        />
      </View>
      <Button
        title={action.pending ? "Đang xử lý..." : "Nhận diện"}
        icon="checkmark-circle-outline"
        loading={action.pending}
        disabled={!photo}
        onPress={() =>
          void action.run(async () => {
            setResult(null);
            setResult(await recognize(server, photo!));
          })
        }
      />
      <RecognitionResultView result={result} />
    </View>
  );
}

const styles = StyleSheet.create({
  modes: { flexDirection: "row", padding: 3, borderRadius: 6, backgroundColor: colors.border },
  mode: {
    flex: 1,
    minHeight: 46,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    borderRadius: 4,
  },
  selected: { backgroundColor: colors.primary },
  preview: {
    width: "100%",
    aspectRatio: 1,
    maxHeight: 400,
    backgroundColor: "#e6eeef",
    borderRadius: 6,
    overflow: "hidden",
    alignItems: "center",
    justifyContent: "center",
  },
  image: { width: "100%", height: "100%" },
});
