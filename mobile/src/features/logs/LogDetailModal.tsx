import { Alert, Modal, ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import type { RecognitionLog } from "../../types";
import { useServer } from "../../providers/ServerProvider";
import { useAction } from "../../hooks/useAction";
import { logService } from "../../services/faceid";
import Button from "../../components/Button";
import Avatar from "../../components/Avatar";
import { ErrorNotice } from "../../components/Feedback";
import { colors, layout } from "../../theme";
import { deviceLabel, personName, recognitionStatusLabel } from "../../utils/labels";

export default function LogDetailModal({
  log,
  onClose,
  onDeleted,
}: {
  log: RecognitionLog;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const { server } = useServer();
  const action = useAction();
  function remove() {
    Alert.alert("Xóa lịch sử?", "Bản ghi và ảnh nhận diện sẽ bị xóa.", [
      { text: "Hủy", style: "cancel" },
      {
        text: "Xóa",
        style: "destructive",
        onPress: () =>
          void action.run(async () => {
            await logService.remove(server, log.id);
            onDeleted();
          }),
      },
    ]);
  }
  return (
    <Modal
      animationType="slide"
      onRequestClose={() => {
        if (!action.pending) onClose();
      }}
    >
      <SafeAreaView style={[layout.fill, { backgroundColor: colors.background }]}>
        <ScrollView contentContainerStyle={layout.content}>
          <Text style={layout.title}>Chi tiết nhận diện</Text>
          <ErrorNotice message={action.error} />
          <Avatar server={server} path={log.imageUrl} size={200} />
          <Text style={layout.subtitle}>{personName(log.name)}</Text>
          {[
            ["Mã nhân viên", log.employeeId === null ? "Chưa xác định" : String(log.employeeId)],
            ["Trạng thái", recognitionStatusLabel[log.status]],
            ["Phòng ban", log.department || "-"],
            ["Thời gian", new Date(log.recognizedAt).toLocaleString("vi-VN")],
            [
              "Ca",
              log.attendancePeriod === "MORNING"
                ? "Sáng"
                : log.attendancePeriod === "AFTERNOON"
                  ? "Chiều"
                  : "-",
            ],
            ["Thiết bị", deviceLabel(log.device)],
            [
              "Độ tương đồng",
              log.similarity == null ? "-" : `${(log.similarity * 100).toFixed(1)}%`,
            ],
            ["Lý do", log.reason || "-"],
            [
              "Ngưỡng nhận diện",
              log.threshold == null ? "-" : `${(log.threshold * 100).toFixed(1)}%`,
            ],
          ].map(([label, value]) => (
            <View key={label} style={{ gap: 4 }}>
              <Text style={layout.muted}>{label}</Text>
              <Text selectable style={layout.text}>
                {value}
              </Text>
            </View>
          ))}
          <View style={layout.actions}>
            <Button
              title="Đóng"
              icon="close-outline"
              variant="secondary"
              disabled={action.pending}
              onPress={onClose}
            />
            <Button
              title="Xóa bản ghi"
              icon="trash-outline"
              variant="danger"
              loading={action.pending}
              onPress={remove}
            />
          </View>
        </ScrollView>
      </SafeAreaView>
    </Modal>
  );
}
