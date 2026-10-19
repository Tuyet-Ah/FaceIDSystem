import { useState } from "react";
import { Image, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { colors } from "../theme";
import { mediaUrl } from "../services/client";

export default function Avatar({
  server,
  path,
  size = 56,
}: {
  server: string;
  path: string | null;
  size?: number;
}) {
  const uri = mediaUrl(server, path);
  const [failedUri, setFailedUri] = useState<string>();
  const style = { width: size, height: size, borderRadius: 6, backgroundColor: colors.primarySoft };
  return uri && failedUri !== uri ? (
    <Image source={{ uri }} style={style} onError={() => setFailedUri(uri)} />
  ) : (
    <View style={[style, { justifyContent: "center", alignItems: "center" }]}>
      <Ionicons name="person-outline" size={size * 0.45} color={colors.primary} />
    </View>
  );
}
