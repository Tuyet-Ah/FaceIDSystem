import { SafeAreaProvider } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import AppNavigator from "./src/navigation/AppNavigator";
import { ServerProvider } from "./src/providers/ServerProvider";
import ErrorBoundary from "./src/components/ErrorBoundary";

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <ErrorBoundary>
        <ServerProvider>
          <AppNavigator />
        </ServerProvider>
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
