import { useCallback, useEffect, useState } from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { useServer } from "../providers/ServerProvider";
import { useResource } from "../hooks/useResource";
import { logService } from "../services/faceid";
import type { RecognitionLog } from "../types";
import Avatar from "../components/Avatar";
import { EmptyState, ErrorNotice, LoadingState } from "../components/Feedback";
import LogDetailModal from "../features/logs/LogDetailModal";
import { colors, layout } from "../theme";
import FormField from "../components/FormField";
import Button from "../components/Button";
import SelectField from "../components/SelectField";
import { filterLogs, type TimeRange } from "../features/logs/filters";
import { personName } from "../utils/labels";

type Filter = "ALL" | "SUCCESS" | "UNKNOWN";
const filters: [Filter, string][] = [
  ["ALL", "Tất cả"],
  ["SUCCESS", "Thành công"],
  ["UNKNOWN", "Không khớp"],
];

export default function LogsScreen({
  initialEmployeeId = null,
}: {
  initialEmployeeId?: number | null;
}) {
  const { server } = useServer();
  const [employeeId, setEmployeeId] = useState(initialEmployeeId);
  useEffect(() => {
    setEmployeeId(initialEmployeeId);
  }, [initialEmployeeId]);
  const resource = useResource(
    useCallback(
      (signal: AbortSignal) =>
        logService.list(server, signal, employeeId === null ? {} : { employeeId }),
      [server, employeeId],
    ),
  );
  const [filter, setFilter] = useState<Filter>("ALL");
  const [query, setQuery] = useState("");
  const [range, setRange] = useState<TimeRange>("all");
  const [selected, setSelected] = useState<RecognitionLog | null>(null);
  const items = filterLogs(resource.data ?? [], { query, status: filter, range, employeeId });

  return (
    <View style={layout.fill}>
      <FlatList
        keyboardShouldPersistTaps="handled"
        data={items}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={layout.content}
        refreshing={resource.loading}
        onRefresh={() => void resource.refresh()}
        ListHeaderComponent={
          <View style={{ gap: 16 }}>
            <Text style={layout.title}>Lịch sử nhận diện</Text>
            <Text style={layout.muted}>500 bản ghi gần nhất</Text>
            <FormField
              label="Tìm kiếm"
              placeholder="Mã nhân viên, tên hoặc phòng ban"
              value={query}
              onChangeText={setQuery}
            />
            {employeeId !== null && (
              <Button
                title={`Nhân viên #${employeeId} — bỏ lọc`}
                icon="close-circle-outline"
                variant="secondary"
                onPress={() => setEmployeeId(null)}
              />
            )}
            <SelectField<TimeRange>
              label="Thời gian"
              value={range}
              onChange={setRange}
              options={[
                { value: "all", label: "Tất cả thời gian" },
                { value: "today", label: "Hôm nay" },
                { value: "yesterday", label: "Hôm qua" },
                { value: "last3days", label: "3 ngày gần đây" },
                { value: "last7days", label: "7 ngày gần đây" },
              ]}
            />
            <View accessibilityRole="tablist" style={styles.filters}>
              {filters.map(([key, label]) => (
                <Pressable
                  accessibilityRole="tab"
                  accessibilityState={{ selected: filter === key }}
                  key={key}
                  onPress={() => setFilter(key)}
                  style={[styles.filter, filter === key && { backgroundColor: colors.primary }]}
                >
                  <Text
                    style={{
                      fontSize: 12,
                      fontWeight: "600",
                      color: filter === key ? "white" : colors.muted,
                    }}
                  >
                    {label}
                  </Text>
                </Pressable>
              ))}
            </View>
            <ErrorNotice message={resource.error} retry={() => void resource.refresh()} />
            <Text style={layout.muted}>{items.length} bản ghi phù hợp</Text>
          </View>
        }
        ListEmptyComponent={
          resource.loading && !resource.data ? (
            <LoadingState />
          ) : resource.error && !resource.data ? null : (
            <EmptyState title="Không có lịch sử" />
          )
        }
        renderItem={({ item }) => (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Xem bản ghi ${personName(item.name)}`}
            onPress={() => setSelected(item)}
            style={styles.row}
          >
            <Avatar server={server} path={item.imageUrl} />
            <View style={layout.fill}>
              <Text style={layout.subtitle}>{personName(item.name)}</Text>
              <Text style={layout.muted}>
                {new Date(item.recognizedAt).toLocaleString("vi-VN")}
              </Text>
              <Text
                style={{
                  color: item.status === "SUCCESS" ? colors.primary : colors.danger,
                  fontSize: 13,
                }}
              >
                {item.status === "SUCCESS" ? "Thành công" : "Không khớp"}
                {item.similarity != null ? ` / ${(item.similarity * 100).toFixed(1)}%` : ""}
              </Text>
            </View>
          </Pressable>
        )}
      />
      {selected && (
        <LogDetailModal
          log={selected}
          onClose={() => setSelected(null)}
          onDeleted={() => {
            setSelected(null);
            void resource.refresh();
          }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  filters: { flexDirection: "row", borderRadius: 6, padding: 3, backgroundColor: colors.border },
  filter: {
    flex: 1,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 4,
    padding: 6,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderColor: colors.border,
  },
});
