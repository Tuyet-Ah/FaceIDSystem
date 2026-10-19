import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Button, { type IconName } from "./Button";
import { colors, layout } from "../theme";

export function ErrorNotice({ message, retry }: { message: string | null; retry?: () => void }) {
  if (!message) return null;
  return (
    <View accessibilityLiveRegion="polite" style={styles.error}>
      <Text style={{ color: colors.danger }}>{message}</Text>
      {retry && (
        <Button title="Thử lại" icon="refresh-outline" variant="secondary" onPress={retry} />
      )}
    </View>
  );
}

export function EmptyState({
  title,
  description,
  icon = "file-tray-outline",
}: {
  title: string;
  description?: string;
  icon?: IconName;
}) {
  return (
    <View style={styles.empty}>
      <Ionicons name={icon} size={40} color={colors.muted} />
      <Text style={layout.subtitle}>{title}</Text>
      {description && <Text style={[layout.muted, { textAlign: "center" }]}>{description}</Text>}
    </View>
  );
}

export function LoadingState() {
  return (
    <View style={styles.empty}>
      <ActivityIndicator color={colors.primary} />
      <Text style={layout.muted}>Đang tải dữ liệu...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  error: { backgroundColor: colors.dangerSoft, borderRadius: 6, padding: 14, gap: 12 },
  empty: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 48,
    paddingHorizontal: 20,
    gap: 12,
  },
});
