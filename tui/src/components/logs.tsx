import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface LogsProps {
  logs: string[];
  theme: Theme;
}

export function Logs({ logs, theme }: LogsProps) {
  return (
    <Box flexDirection="column" flexGrow={1} overflow="hidden">
      <Text bold color={theme.accent}>
        Logs
      </Text>
      {logs.slice(-20).map((line, idx) => (
        <Text key={idx} color={theme.muted}>
          {line}
        </Text>
      ))}
    </Box>
  );
}
