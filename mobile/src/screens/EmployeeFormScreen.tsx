import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  BackHandler,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Text,
  View,
} from "react-native";
import type { ImagePickerAsset } from "expo-image-picker";
import type { Employee, ImageWarning } from "../types";
import { useServer } from "../providers/ServerProvider";
import { useAction } from "../hooks/useAction";
import { employeeService } from "../services/faceid";
import { ApiError } from "../services/client";
import { pickImages } from "../utils/images";
import { emptyEmployee, employeeFields, validateEmployee } from "../features/employees/validation";
import PhotoPicker from "../features/employees/PhotoPicker";
import FormField from "../components/FormField";
import Button from "../components/Button";
import { ErrorNotice } from "../components/Feedback";
import { layout } from "../theme";

export default function EmployeeFormScreen({
  employee,
  onClose,
  onSaved,
}: {
  employee?: Employee;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { server } = useServer();
  const initial = useMemo(
    () => ({
      ...emptyEmployee,
      ...Object.fromEntries(employeeFields.map(({ key }) => [key, employee?.[key] ?? ""])),
    }),
    [employee],
  );
  const [form, setForm] = useState(initial);
  const [photos, setPhotos] = useState<ImagePickerAsset[]>([]);
  const [primary, setPrimary] = useState(0);
  const [submitted, setSubmitted] = useState(false);
  const [imageErrors, setImageErrors] = useState<ImageWarning[]>([]);
  const action = useAction();
  const scroll = useRef<ScrollView>(null);
  const errors = validateEmployee(form, photos.length, !employee);
  const dirty = photos.length > 0 || employeeFields.some(({ key }) => form[key] !== initial[key]);

  function close() {
    if (action.pending) return;
    if (!dirty) {
      onClose();
      return;
    }
    Alert.alert("Bỏ thay đổi?", "Thông tin chưa lưu sẽ bị mất.", [
      { text: "Tiếp tục sửa", style: "cancel" },
      { text: "Bỏ thay đổi", style: "destructive", onPress: onClose },
    ]);
  }

  useEffect(() => {
    const subscription = BackHandler.addEventListener("hardwareBackPress", () => {
      close();
      return true;
    });
    return () => subscription.remove();
  }, [dirty, action.pending, onClose]);

  async function save() {
    setSubmitted(true);
    if (Object.keys(errors).length) {
      scroll.current?.scrollTo({ y: 0, animated: true });
      return;
    }
    await action.run(async () => {
      setImageErrors([]);
      try {
        const saved = await employeeService.save(server, form, photos, primary, employee?.id);
        if (saved.imageWarnings?.length)
          Alert.alert(
            "Đã lưu - một số ảnh bị loại",
            saved.imageWarnings.map((item) => `${item.fileName}: ${item.message}`).join("\n"),
          );
        onSaved();
      } catch (e) {
        if (e instanceof ApiError) setImageErrors(e.imageErrors);
        throw e;
      }
    });
  }

  return (
    <KeyboardAvoidingView
      style={layout.fill}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        ref={scroll}
        contentContainerStyle={layout.content}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={layout.title}>{employee ? "Sửa nhân viên" : "Thêm nhân viên"}</Text>
        <ErrorNotice message={action.error} />
        {employeeFields.map(({ key, label }) => (
          <FormField
            key={key}
            label={label}
            value={form[key]}
            editable={!action.pending}
            error={submitted ? errors[key] : undefined}
            onChangeText={(value) => setForm((current) => ({ ...current, [key]: value }))}
            autoCapitalize={key === "email" ? "none" : "sentences"}
            autoCorrect={key !== "email" && key !== "phone"}
            keyboardType={
              key === "email"
                ? "email-address"
                : key === "phone"
                  ? "phone-pad"
                  : key === "dateOfBirth"
                    ? "numbers-and-punctuation"
                    : "default"
            }
          />
        ))}
        {employee && !photos.length && (
          <Text style={layout.muted}>Bộ ảnh hiện tại: {employee.faceImageCount} ảnh</Text>
        )}
        <PhotoPicker
          photos={photos}
          primary={primary}
          disabled={action.pending}
          error={submitted ? errors.images : undefined}
          onChoose={() =>
            void action.run(async () => {
              const selected = await pickImages(false, true);
              if (selected) {
                setPhotos(selected);
                setPrimary(0);
                setImageErrors([]);
              }
            })
          }
          onPrimary={setPrimary}
          onRemove={(index) => {
            setPhotos((current) => current.filter((_, i) => i !== index));
            setPrimary((current) =>
              current === index ? 0 : current > index ? current - 1 : current,
            );
            setImageErrors([]);
          }}
        />
        {imageErrors.map((item, index) => (
          <ErrorNotice key={index} message={`${item.fileName}: ${item.message}`} />
        ))}
        <View style={layout.actions}>
          <Button
            title="Lưu nhân viên"
            icon="save-outline"
            loading={action.pending}
            onPress={() => void save()}
          />
          <Button
            title="Hủy"
            icon="close-outline"
            variant="secondary"
            disabled={action.pending}
            onPress={close}
          />
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
