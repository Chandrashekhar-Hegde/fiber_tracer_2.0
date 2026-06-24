import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface DashboardProps {
  summary?: Record<string, unknown>;
  theme: Theme;
}

export function Dashboard({ summary, theme }: DashboardProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Dashboard
      </Text>
      <Text color={theme.foreground}>Fibers: {String(summary?.n_labels ?? "-")}</Text>
      <Text color={theme.foreground}>Regime: {String(summary?.regime ?? "-")}</Text>
      <Text color={theme.foreground}>Elapsed: {String(summary?.elapsed_seconds ?? "-")}s</Text>
      <Text color={theme.muted}>Fiber table and charts coming in Task 10.</Text>
    </Box>
  );
}
