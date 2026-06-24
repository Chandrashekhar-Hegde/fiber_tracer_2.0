import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface TrainingProps {
  theme: Theme;
}

export function Training({ theme }: TrainingProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Training</Text>
      <Text color={theme.foreground}>Task: Semantic segmentation fine-tuning</Text>
      <Text color={theme.foreground}>Model: unet-v3.2 (frozen encoder)</Text>
      <Text color={theme.foreground}>Epochs: 0 / 50</Text>
      <Text color={theme.muted}>Start training from the CLI with: fiber-tracer train --dataset-dir ./data (v3.3+).</Text>
    </Box>
  );
}
