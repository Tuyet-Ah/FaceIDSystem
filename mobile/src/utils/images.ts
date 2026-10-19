import * as ImagePicker from "expo-image-picker";
import { Alert, Linking } from "react-native";

export async function pickImages(camera: boolean, multiple = false) {
  if (camera) {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Cần quyền camera", "Cho phép camera trong cài đặt để chụp ảnh điểm danh.", [
        { text: "Đóng", style: "cancel" },
        {
          text: "Mở cài đặt",
          onPress: () => {
            void Linking.openSettings();
          },
        },
      ]);
      return null;
    }
  }
  const options: ImagePicker.ImagePickerOptions = {
    mediaTypes: ["images"],
    quality: 0.9,
    allowsMultipleSelection: multiple,
    selectionLimit: multiple ? 10 : 1,
  };
  const result = camera
    ? await ImagePicker.launchCameraAsync(options)
    : await ImagePicker.launchImageLibraryAsync(options);
  if (result.canceled) return null;
  for (const image of result.assets) {
    if (image.mimeType && !["image/jpeg", "image/png", "image/webp"].includes(image.mimeType)) {
      throw new Error("Chỉ hỗ trợ ảnh JPEG, PNG và WebP");
    }
    if (image.fileSize && image.fileSize > 10 * 1024 * 1024)
      throw new Error("Mỗi ảnh tối đa 10 MB");
  }
  if (result.assets.reduce((total, image) => total + (image.fileSize ?? 0), 0) > 58 * 1024 * 1024) {
    throw new Error("Tổng dung lượng ảnh tối đa 58 MB");
  }
  return result.assets;
}
