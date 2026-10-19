import { useCallback, useMemo, useState } from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useServer } from "../providers/ServerProvider";
import { employeeService } from "../services/faceid";
import { useResource } from "../hooks/useResource";
import type { Employee } from "../types";
import Avatar from "../components/Avatar";
import Button from "../components/Button";
import FormField from "../components/FormField";
import { EmptyState, ErrorNotice, LoadingState } from "../components/Feedback";
import { colors, layout } from "../theme";
import { normalizeSearch } from "../features/logs/filters";

export default function EmployeesScreen({
  onNew,
  onSelect,
}: {
  onNew: () => void;
  onSelect: (employee: Employee) => void;
}) {
  const { server } = useServer();
  const resource = useResource(
    useCallback((signal: AbortSignal) => employeeService.list(server, signal), [server]),
  );
  const [search, setSearch] = useState("");
  const items = useMemo(() => {
    const query = normalizeSearch(search);
    return (resource.data ?? []).filter((employee) =>
      normalizeSearch(
        `${employee.id} ${employee.name} ${employee.email} ${employee.department}`,
      ).includes(query),
    );
  }, [resource.data, search]);

  return (
    <FlatList
      data={items}
      keyExtractor={(item) => String(item.id)}
      contentContainerStyle={layout.content}
      refreshing={resource.loading}
      onRefresh={() => void resource.refresh()}
      keyboardShouldPersistTaps="handled"
      ListHeaderComponent={
        <View style={{ gap: 16 }}>
          <View style={layout.row}>
            <View style={layout.fill}>
              <Text style={layout.title}>Nhân viên</Text>
              <Text style={layout.muted}>{resource.data?.length ?? 0} hồ sơ</Text>
            </View>
            <Button title="Thêm" icon="add" onPress={onNew} />
          </View>
          <FormField
            label="Tìm kiếm"
            placeholder="Tên, email, phòng ban"
            value={search}
            onChangeText={setSearch}
          />
          <ErrorNotice message={resource.error} retry={() => void resource.refresh()} />
        </View>
      }
      ListEmptyComponent={
        resource.loading && !resource.data ? (
          <LoadingState />
        ) : resource.error && !resource.data ? null : (
          <EmptyState
            title={search ? "Không tìm thấy nhân viên" : "Chưa có nhân viên"}
            icon="people-outline"
          />
        )
      }
      renderItem={({ item }) => (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Xem ${item.name}`}
          onPress={() => onSelect(item)}
          style={({ pressed }) => [styles.row, pressed && { backgroundColor: colors.primarySoft }]}
        >
          <Avatar server={server} path={item.primaryPhotoUrl} />
          <View style={layout.fill}>
            <Text style={layout.subtitle}>{item.name}</Text>
            <Text style={layout.muted}>{item.department}</Text>
            <Text numberOfLines={1} style={layout.muted}>
              {item.email}
            </Text>
            <Text
              style={{
                color: item.faceStatus === "READY" ? colors.primary : colors.danger,
                fontSize: 12,
              }}
            >
              {item.faceStatus === "READY"
                ? "Sẵn sàng nhận diện"
                : item.faceStatus === "PROCESSING"
                  ? "Đang xử lý"
                  : "Lỗi dữ liệu khuôn mặt"}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color={colors.muted} />
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderColor: colors.border,
  },
});
