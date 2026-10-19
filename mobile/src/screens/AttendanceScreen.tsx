import { useCallback, useState } from "react";
import { FlatList, StyleSheet, Text, View } from "react-native";
import { useServer } from "../providers/ServerProvider";
import { useResource } from "../hooks/useResource";
import { employeeService, logService } from "../services/faceid";
import { todayCheckIns } from "../features/logs/filters";
import Avatar from "../components/Avatar";
import SelectField from "../components/SelectField";
import { EmptyState, ErrorNotice, LoadingState } from "../components/Feedback";
import { colors, layout } from "../theme";

type Period = "ALL" | "MORNING" | "AFTERNOON";

export default function AttendanceScreen() {
  const { server } = useServer();
  const employees = useResource(
    useCallback((signal: AbortSignal) => employeeService.list(server, signal), [server]),
    30_000,
  );
  const logs = useResource(
    useCallback(
      (signal: AbortSignal) => logService.list(server, signal, { status: "SUCCESS" }),
      [server],
    ),
    5_000,
  );
  const [period, setPeriod] = useState<Period>("ALL");
  const [refreshing, setRefreshing] = useState(false);
  const checkIns = todayCheckIns(logs.data ?? []);
  const items = checkIns.filter((log) => period === "ALL" || log.attendancePeriod === period);
  const stats: [string, number | undefined][] = [
    ["Sẵn sàng", employees.data?.filter((employee) => employee.faceStatus === "READY").length],
    [
      "Ca sáng",
      logs.data ? checkIns.filter((log) => log.attendancePeriod === "MORNING").length : undefined,
    ],
    [
      "Ca chiều",
      logs.data ? checkIns.filter((log) => log.attendancePeriod === "AFTERNOON").length : undefined,
    ],
  ];

  return (
    <FlatList
      data={items}
      keyExtractor={(log) => `${log.employeeId}:${log.attendancePeriod}`}
      contentContainerStyle={layout.content}
      refreshing={refreshing}
      onRefresh={() => {
        setRefreshing(true);
        void Promise.all([logs.refresh(), employees.refresh()]).finally(() => setRefreshing(false));
      }}
      ListHeaderComponent={
        <View style={{ gap: 16 }}>
          <Text style={layout.title}>Điểm danh hôm nay</Text>
          <Text style={layout.muted}>{new Date().toLocaleDateString("vi-VN")}</Text>
          <View style={styles.stats}>
            {stats.map(([label, value]) => (
              <View key={label} style={styles.stat}>
                <Text style={layout.muted}>{label}</Text>
                <Text style={styles.count}>{value ?? "—"}</Text>
              </View>
            ))}
          </View>
          <ErrorNotice message={employees.error} retry={() => void employees.refresh()} />
          <ErrorNotice message={logs.error} retry={() => void logs.refresh()} />
          <SelectField<Period>
            label="Ca làm việc"
            value={period}
            onChange={setPeriod}
            options={[
              { value: "ALL", label: "Tất cả các ca" },
              { value: "MORNING", label: "Ca sáng" },
              { value: "AFTERNOON", label: "Ca chiều" },
            ]}
          />
          <Text style={layout.muted}>
            {items.length} lượt hôm nay trong 500 bản ghi thành công gần nhất
          </Text>
        </View>
      }
      ListEmptyComponent={
        !logs.data && logs.loading ? (
          <LoadingState />
        ) : logs.error && !logs.data ? null : (
          <EmptyState title="Chưa có lượt điểm danh" icon="checkmark-done-outline" />
        )
      }
      renderItem={({ item }) => (
        <View style={styles.row}>
          <Avatar
            server={server}
            path={
              employees.data?.find((employee) => employee.id === item.employeeId)
                ?.primaryPhotoUrl ?? item.imageUrl
            }
          />
          <View style={layout.fill}>
            <Text style={layout.subtitle}>{item.name}</Text>
            <Text style={layout.muted}>{item.department}</Text>
            <Text style={{ color: colors.primary }}>
              {item.attendancePeriod === "MORNING" ? "Ca sáng" : "Ca chiều"} /{" "}
              {new Date(item.recognizedAt).toLocaleTimeString("vi-VN")}
            </Text>
          </View>
        </View>
      )}
    />
  );
}

const styles = StyleSheet.create({
  stats: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderColor: colors.border,
    paddingBottom: 16,
    gap: 10,
  },
  stat: { flex: 1, gap: 6 },
  count: { fontSize: 26, fontWeight: "700", color: colors.primary },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderColor: colors.border,
  },
});
