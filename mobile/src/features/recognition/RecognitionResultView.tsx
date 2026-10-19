import { Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type { RecognitionResult } from "../../types";
import { colors, layout } from "../../theme";

export default function RecognitionResultView({ result }: { result: RecognitionResult | null }) {
  if (!result) return null;
  return (
    <View
      accessibilityLiveRegion="polite"
      style={{
        padding: 20,
        gap: 10,
        borderRadius: 6,
        backgroundColor: result.matched ? colors.primarySoft : colors.dangerSoft,
      }}
    >
      <Ionicons
        name={result.matched ? "checkmark-circle" : "help-circle"}
        size={32}
        color={result.matched ? colors.primary : colors.danger}
      />
      <Text style={layout.subtitle}>{result.employee?.name ?? "Không nhận diện được"}</Text>
      {!!result.employee?.department && (
        <Text style={layout.text}>{result.employee.department}</Text>
      )}
      <Text style={layout.text}>{result.message}</Text>
      {result.similarity != null && (
        <Text style={layout.muted}>Độ tương đồng: {(result.similarity * 100).toFixed(1)}%</Text>
      )}
      {result.duplicate && <Text style={layout.muted}>Ca hiện tại đã được ghi nhận trước đó.</Text>}
    </View>
  );
}
