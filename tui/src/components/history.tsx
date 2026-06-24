import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { RunRecord } from "../types";

interface HistoryProps {
  history: RunRecord[];
  theme: Theme;
}

export function History({ history, theme }: HistoryProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        History
      </Text>
      {history.length === 0 && <Text color={theme.muted}>No runs yet.</Text>}
      {history.map((run) => (
        <Text key={run.id} color={theme.foreground}>
          {run.status === "success" ? "✓" : "✗"} {run.name} — {run.startedAt}
        </Text>
      ))}
    </Box>
  );
}
