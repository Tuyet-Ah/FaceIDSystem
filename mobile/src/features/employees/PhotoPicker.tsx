import { Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import type { ImagePickerAsset } from "expo-image-picker";
import { Ionicons } from "@expo/vector-icons";
import Button from "../../components/Button";
import { ErrorNotice } from "../../components/Feedback";
import { colors, layout } from "../../theme";

interface Props {
  photos: ImagePickerAsset[];
  primary: number;
  disabled: boolean;
  error?: string;
  onChoose: () => void;
  onPrimary: (index: number) => void;
  onRemove: (index: number) => void;
}

export default function PhotoPicker({
  photos,
  primary,
  disabled,
  error,
  onChoose,
  onPrimary,
  onRemove,
}: Props) {
  return (
    <View style={{ gap: 12 }}>
      <View style={layout.row}>
        <Text style={[layout.subtitle, layout.fill]}>Ảnh khuôn mặt</Text>
        <Text style={layout.muted}>{photos.length}/10</Text>
      </View>
      <ErrorNotice message={error ?? null} />
      {!!photos.length && (
        <ScrollView horizontal contentContainerStyle={{ gap: 10 }}>
          {photos.map((photo, index) => (
            <View key={`${photo.uri}-${index}`} style={{ width: 92 }}>
              <Pressable
                disabled={disabled}
                accessibilityRole="radio"
                accessibilityState={{ checked: primary === index }}
                accessibilityLabel={`Ảnh ${index + 1}, chọn làm đại diện`}
                onPress={() => onPrimary(index)}
              >
                <Image
                  source={{ uri: photo.uri }}
                  style={[styles.photo, primary === index && { borderColor: colors.primary }]}
                />
                <Text style={styles.caption}>
                  {primary === index ? "Đại diện" : `Ảnh ${index + 1}`}
                </Text>
              </Pressable>
              <Pressable
                disabled={disabled}
                accessibilityRole="button"
                accessibilityLabel={`Bỏ ảnh ${index + 1}`}
                onPress={() => onRemove(index)}
                style={styles.remove}
              >
                <Ionicons name="close-circle" size={24} color={colors.danger} />
              </Pressable>
            </View>
          ))}
        </ScrollView>
      )}
      <Button
        title={photos.length ? "Thay bộ ảnh" : "Chọn ảnh"}
        icon="images-outline"
        variant="secondary"
        disabled={disabled}
        onPress={onChoose}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  photo: { width: 92, height: 112, borderRadius: 6, borderWidth: 3, borderColor: colors.border },
  caption: { fontSize: 12, color: colors.primary, paddingVertical: 6, textAlign: "center" },
  remove: { minHeight: 44, alignItems: "center", justifyContent: "center" },
});
