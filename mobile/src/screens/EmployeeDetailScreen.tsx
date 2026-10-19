import { useCallback } from "react";
import { Alert, ScrollView, Text, View } from "react-native";
import type { Employee } from "../types";
import { useServer } from "../providers/ServerProvider";
import { useResource } from "../hooks/useResource";
import { useAction } from "../hooks/useAction";
import { employeeService } from "../services/faceid";
import { employeeFields } from "../features/employees/validation";
import Avatar from "../components/Avatar";
import Button from "../components/Button";
import { ErrorNotice, LoadingState } from "../components/Feedback";
import { layout } from "../theme";
import { faceStatusLabel } from "../utils/labels";

export default function EmployeeDetailScreen({
  id,
  onEdit,
  onDeleted,
  onHistory,
}: {
  id: number;
  onEdit: (employee: Employee) => void;
  onDeleted: () => void;
  onHistory: () => void;
}) {
  const { server } = useServer();
  const resource = useResource(
    useCallback((signal: AbortSignal) => employeeService.get(server, id, signal), [server, id]),
  );
  const action = useAction();
  const employee = resource.data;

  function remove() {
    Alert.alert("Xóa nhân viên?", "Hồ sơ, ảnh đăng ký và dữ liệu khuôn mặt sẽ bị xóa.", [
      { text: "Hủy", style: "cancel" },
      {
        text: "Xóa",
        style: "destructive",
        onPress: () =>
          void action.run(async () => {
            await employeeService.remove(server, id);
            onDeleted();
          }),
      },
    ]);
  }

  return (
    <ScrollView contentContainerStyle={layout.content}>
      <ErrorNotice message={resource.error} retry={() => void resource.refresh()} />
      <ErrorNotice message={action.error} />
      {!employee && resource.loading && <LoadingState />}
      {employee && (
        <>
          <View style={layout.row}>
            <Avatar server={server} path={employee.primaryPhotoUrl} size={88} />
            <View style={layout.fill}>
              <Text style={layout.title}>{employee.name}</Text>
              <Text style={layout.muted}>
                #{employee.id} / {faceStatusLabel[employee.faceStatus]}
              </Text>
              <Text style={layout.muted}>{employee.faceImageCount} ảnh đăng ký</Text>
            </View>
          </View>
          <View style={layout.divider} />
          {employeeFields.map(({ key, label }) => (
            <View key={key} style={{ gap: 4 }}>
              <Text style={layout.muted}>{label}</Text>
              <Text selectable style={layout.text}>
                {employee[key]}
              </Text>
            </View>
          ))}
          <View style={layout.actions}>
            <Button
              title="Lịch sử điểm danh"
              icon="time-outline"
              variant="secondary"
              disabled={action.pending}
              onPress={onHistory}
            />
            <Button
              title="Sửa hồ sơ"
              icon="create-outline"
              disabled={action.pending}
              onPress={() => onEdit(employee)}
            />
            <Button
              title="Xóa nhân viên"
              icon="trash-outline"
              variant="danger"
              loading={action.pending}
              onPress={remove}
            />
          </View>
        </>
      )}
    </ScrollView>
  );
}
