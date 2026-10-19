import { useState } from "react";
import { FlatList, Modal, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import Button from "./Button";
import { colors, layout } from "../theme";

export default function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <View style={{ gap: 7 }}>
      <Text style={layout.muted}>{label}</Text>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`${label}: ${options.find((option) => option.value === value)?.label}`}
        onPress={() => setOpen(true)}
        style={styles.control}
      >
        <Text style={[layout.text, layout.fill]}>
          {options.find((option) => option.value === value)?.label}
        </Text>
        <Ionicons name="chevron-down" size={20} color={colors.muted} />
      </Pressable>
      <Modal visible={open} animationType="slide" onRequestClose={() => setOpen(false)}>
        <SafeAreaView style={[layout.fill, { backgroundColor: colors.background }]}>
          <FlatList
            data={options}
            keyExtractor={(item) => item.value}
            contentContainerStyle={layout.content}
            ListHeaderComponent={<Text style={layout.title}>{label}</Text>}
            renderItem={({ item }) => (
              <Pressable
                accessibilityRole="radio"
                accessibilityState={{ checked: value === item.value }}
                style={styles.option}
                onPress={() => {
                  onChange(item.value);
                  setOpen(false);
                }}
              >
                <Text style={[layout.text, layout.fill]}>{item.label}</Text>
                <Ionicons
                  name={value === item.value ? "radio-button-on" : "radio-button-off"}
                  size={24}
                  color={colors.primary}
                />
              </Pressable>
            )}
            ListFooterComponent={
              <Button
                title="Đóng"
                icon="close-outline"
                variant="secondary"
                onPress={() => setOpen(false)}
              />
            }
          />
        </SafeAreaView>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  control: {
    minHeight: 48,
    padding: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    borderRadius: 6,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  option: {
    minHeight: 56,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    borderBottomWidth: 1,
    borderColor: colors.border,
    paddingVertical: 12,
  },
});
