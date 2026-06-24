import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface ModelRegistryProps {
  theme: Theme;
}

export function ModelRegistry({ theme }: ModelRegistryProps) {
  const models = [
    { id: "unet-v3.2", source: "bundled", status: "ready" },
    { id: "unet-custom-001", source: "local", status: "ready" },
    { id: "swin-fiber-v1", source: "remote", status: "planned" },
  ];
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Model Registry</Text>
      {models.map((m) => (
        <Text key={m.id} color={theme.foreground}>
          • {m.id} ({m.source}) — {m.status}
        </Text>
      ))}
      <Text color={theme.muted}>Import, export, and version models here (v3.3+).</Text>
    </Box>
  );
}
