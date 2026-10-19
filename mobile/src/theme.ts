import { StyleSheet } from "react-native";

export const colors = {
  background: "#f5f7f9",
  surface: "#ffffff",
  text: "#20333b",
  muted: "#657780",
  border: "#dce4e8",
  primary: "#087f70",
  primarySoft: "#e2f3ee",
  danger: "#b42336",
  dangerSoft: "#fff0f1",
  accent: "#2968ad",
  accentSoft: "#eaf1fa",
};

export const layout = StyleSheet.create({
  fill: { flex: 1 },
  content: {
    padding: 20,
    paddingBottom: 40,
    gap: 16,
    width: "100%",
    maxWidth: 760,
    alignSelf: "center",
  },
  title: { fontSize: 24, fontWeight: "700", color: colors.text },
  subtitle: { fontSize: 17, fontWeight: "700", color: colors.text },
  text: { fontSize: 15, lineHeight: 23, color: colors.text },
  muted: { fontSize: 13, lineHeight: 20, color: colors.muted },
  row: { flexDirection: "row", alignItems: "center", gap: 12 },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  divider: { height: 1, backgroundColor: colors.border },
});
