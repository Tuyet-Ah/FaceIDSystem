import { useEffect, useState } from "react";
import { BackHandler, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useServer } from "../providers/ServerProvider";
import type { Employee } from "../types";
import type { IconName } from "../components/Button";
import { LoadingState } from "../components/Feedback";
import EmployeesScreen from "../screens/EmployeesScreen";
import EmployeeDetailScreen from "../screens/EmployeeDetailScreen";
import EmployeeFormScreen from "../screens/EmployeeFormScreen";
import RecognitionScreen from "../screens/RecognitionScreen";
import LogsScreen from "../screens/LogsScreen";
import SettingsScreen from "../screens/SettingsScreen";
import AttendanceScreen from "../screens/AttendanceScreen";
import { colors, layout } from "../theme";

type Tab = "employees" | "recognition" | "attendance" | "logs" | "settings";
type Route =
  { name: "list" } | { name: "detail"; id: number } | { name: "form"; employee?: Employee };
const tabs: { key: Tab; label: string; icon: IconName }[] = [
  { key: "employees", label: "Nhân viên", icon: "people-outline" },
  { key: "recognition", label: "Điểm danh", icon: "scan-outline" },
  { key: "attendance", label: "Hôm nay", icon: "checkmark-done-outline" },
  { key: "logs", label: "Lịch sử", icon: "time-outline" },
  { key: "settings", label: "Cài đặt", icon: "settings-outline" },
];

export default function AppNavigator() {
  const { ready, server } = useServer();
  const [tab, setTab] = useState<Tab>("employees");
  const [route, setRoute] = useState<Route>({ name: "list" });
  const [historyEmployeeId, setHistoryEmployeeId] = useState<number | null>(null);
  const formOpen = tab === "employees" && route.name === "form";

  useEffect(() => {
    if (formOpen) return;
    const subscription = BackHandler.addEventListener("hardwareBackPress", () => {
      if (route.name !== "list") {
        setRoute({ name: "list" });
        return true;
      }
      if (tab !== "employees") {
        setTab("employees");
        return true;
      }
      return false;
    });
    return () => subscription.remove();
  }, [tab, route.name, formOpen]);

  function closeForm() {
    setRoute(
      route.name === "form" && route.employee
        ? { name: "detail", id: route.employee.id }
        : { name: "list" },
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <View style={layout.fill}>
          <Text style={styles.brand}>FaceID</Text>
          <Text style={layout.muted}>Quản lý nhân viên & điểm danh</Text>
        </View>
        {route.name === "detail" && tab === "employees" ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Quay lại danh sách"
            onPress={() => setRoute({ name: "list" })}
            style={styles.icon}
          >
            <Ionicons name="arrow-back" size={25} color={colors.primary} />
          </Pressable>
        ) : (
          <Ionicons name="finger-print-outline" size={32} color={colors.primary} />
        )}
      </View>
      {!ready ? (
        <LoadingState />
      ) : (
        <View style={layout.fill} key={tab === "settings" ? "settings" : `${server}:${tab}`}>
          {tab === "employees" &&
            (route.name === "form" ? (
              <EmployeeFormScreen
                employee={route.employee}
                onClose={closeForm}
                onSaved={() => setRoute({ name: "list" })}
              />
            ) : route.name === "detail" ? (
              <EmployeeDetailScreen
                id={route.id}
                onEdit={(employee) => setRoute({ name: "form", employee })}
                onDeleted={() => setRoute({ name: "list" })}
                onHistory={() => {
                  setHistoryEmployeeId(route.id);
                  setRoute({ name: "list" });
                  setTab("logs");
                }}
              />
            ) : (
              <EmployeesScreen
                onNew={() => setRoute({ name: "form" })}
                onSelect={(employee) => setRoute({ name: "detail", id: employee.id })}
              />
            ))}
          {tab === "recognition" && <RecognitionScreen />}
          {tab === "attendance" && <AttendanceScreen />}
          {tab === "logs" && <LogsScreen initialEmployeeId={historyEmployeeId} />}
          {tab === "settings" && <SettingsScreen />}
        </View>
      )}
      {!formOpen && (
        <View style={styles.tabs}>
          {tabs.map((item) => (
            <Pressable
              key={item.key}
              accessibilityRole="tab"
              accessibilityState={{ selected: tab === item.key }}
              disabled={!ready}
              onPress={() => {
                setTab(item.key);
                setHistoryEmployeeId(null);
                setRoute({ name: "list" });
              }}
              style={[styles.tab, tab === item.key && styles.activeTab]}
            >
              <Ionicons
                name={item.icon}
                size={23}
                color={tab === item.key ? colors.primary : colors.muted}
              />
              <Text
                style={[
                  styles.tabLabel,
                  tab === item.key && { color: colors.primary, fontWeight: "700" },
                ]}
              >
                {item.label}
              </Text>
            </Pressable>
          ))}
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 14,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderColor: colors.border,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  brand: { fontSize: 25, fontWeight: "800", color: colors.text },
  icon: { minWidth: 48, minHeight: 48, alignItems: "center", justifyContent: "center" },
  tabs: {
    flexDirection: "row",
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  tab: {
    flex: 1,
    minHeight: 68,
    justifyContent: "center",
    alignItems: "center",
    gap: 6,
    borderTopWidth: 2,
    borderColor: "transparent",
    paddingHorizontal: 3,
  },
  activeTab: { borderColor: colors.primary },
  tabLabel: { fontSize: 11, color: colors.muted, textAlign: "center" },
});
