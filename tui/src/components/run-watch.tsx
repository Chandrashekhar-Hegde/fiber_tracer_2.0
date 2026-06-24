import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { BridgeStatus, ProgressEvent } from "../types";

interface RunWatchProps {
  status: BridgeStatus;
  progress: ProgressEvent | null;
  logs: string[];
  error: string | null;
  theme: Theme;
}

export function RunWatch({ status, progress, logs, error, theme }: RunWatchProps) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text bold color={theme.accent}>
        4. Run & Watch
      </Text>
      <Text color={theme.foreground}>Status: {status}</Text>
      {progress && (
        <>
          <Text color={theme.foreground}>
            {progress.stage} — {progress.percent}%
          </Text>
          <Text color={theme.muted}>{progress.message}</Text>
        </>
      )}
      <Box flexDirection="column" marginTop={1} flexGrow={1} overflow="hidden">
        {logs.slice(-5).map((line, idx) => (
          <Text key={idx} color={theme.muted}>
            {line}
          </Text>
        ))}
      </Box>
      {error && (
        <Box marginTop={1}>
          <Text color={theme.error}>Error: {error}</Text>
        </Box>
      )}
    </Box>
  );
}
