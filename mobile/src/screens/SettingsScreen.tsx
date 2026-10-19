import { useEffect, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useServer } from "../providers/ServerProvider";
import { useAction } from "../hooks/useAction";
import Button from "../components/Button";
import FormField from "../components/FormField";
import { ErrorNotice } from "../components/Feedback";
import { colors, layout } from "../theme";
import ServerCameraStatus from "../features/settings/ServerCameraStatus";

export default function SettingsScreen() {
  const { server, connect, storageError } = useServer();
  const [input, setInput] = useState(server);
  const [connected, setConnected] = useState(false);
  const action = useAction();
  useEffect(() => {
    setInput(server);
  }, [server]);

  return (
    <ScrollView contentContainerStyle={layout.content} keyboardShouldPersistTaps="handled">
      <Text style={layout.title}>Kết nối máy chủ</Text>
      <ErrorNotice message={storageError} />
      <ErrorNotice message={action.error} />
      <FormField
        label="Địa chỉ máy chủ"
        value={input}
        onChangeText={(value) => {
          setInput(value);
          setConnected(false);
          action.clearError();
        }}
        editable={!action.pending}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="http://192.168.1.10:5000"
      />
      <Button
        title="Lưu và kết nối"
        icon="link-outline"
        loading={action.pending}
        onPress={() =>
          void action.run(async () => {
            setConnected(false);
            await connect(input);
            setConnected(true);
          })
        }
      />
      {connected && (
        <View style={layout.row}>
          <Ionicons name="checkmark-circle" size={22} color={colors.primary} />
          <Text style={{ color: colors.primary }}>Kết nối thành công</Text>
        </View>
      )}
      <View style={layout.divider} />
      <Text style={layout.muted}>Máy chủ đã lưu</Text>
      <Text selectable style={layout.text}>
        {server}
      </Text>
      <View style={layout.divider} />
      <ServerCameraStatus />
      <Text style={layout.muted}>FaceID / 1.0.0</Text>
    </ScrollView>
  );
}
