import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface ExperimentsProps {
  theme: Theme;
}

export function Experiments({ theme }: ExperimentsProps) {
  const experiments = [
    { id: "exp-001", name: "Default U-Net baseline", status: "completed" },
    { id: "exp-002", name: "LoRA fine-tune on HT3", status: "planned" },
  ];
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Experiments</Text>
      {experiments.map((e) => (
        <Text key={e.id} color={theme.foreground}>
          • {e.name} — {e.status}
        </Text>
      ))}
      <Text color={theme.muted}>Track hyper-parameter sweeps and compare runs (v3.3+).</Text>
    </Box>
  );
}
