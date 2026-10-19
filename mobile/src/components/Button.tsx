import { ActivityIndicator, Pressable, StyleSheet, Text } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { colors } from "../theme";

export type IconName = keyof typeof Ionicons.glyphMap;
interface Props {
  title: string;
  icon: IconName;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: "primary" | "secondary" | "danger";
}

export default function Button({
  title,
  icon,
  onPress,
  disabled,
  loading,
  variant = "primary",
}: Props) {
  const foreground = variant === "secondary" ? colors.primary : "white";
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!disabled || !!loading, busy: !!loading }}
      disabled={disabled || loading}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        {
          backgroundColor:
            variant === "secondary"
              ? colors.primarySoft
              : variant === "danger"
                ? colors.danger
                : colors.primary,
        },
        (disabled || loading) && styles.disabled,
        pressed && { opacity: 0.8 },
      ]}
    >
      {loading ? (
        <ActivityIndicator color={foreground} />
      ) : (
        <Ionicons name={icon} size={20} color={foreground} />
      )}
      <Text style={[styles.text, { color: foreground }]}>{title}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 48,
    borderRadius: 6,
    paddingVertical: 12,
    paddingHorizontal: 16,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 8,
  },
  text: { fontSize: 14, fontWeight: "700", flexShrink: 1 },
  disabled: { opacity: 0.5 },
});
