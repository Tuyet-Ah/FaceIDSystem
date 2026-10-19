import { Component, type ReactNode } from "react";
import { View, Text } from "react-native";
import { layout } from "../theme";
import Button from "./Button";

export default class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <View style={[layout.fill, layout.content, { justifyContent: "center" }]}>
          <Text style={layout.title}>Không thể hiển thị ứng dụng</Text>
          <Text style={layout.text}>Vui lòng thử lại.</Text>
          <Button
            title="Thử lại"
            icon="refresh-outline"
            onPress={() => this.setState({ failed: false })}
          />
        </View>
      );
    return this.props.children;
  }
}
