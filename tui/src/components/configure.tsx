import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { AnalysisConfig } from "../types";

interface ConfigureProps {
  config: AnalysisConfig;
  theme: Theme;
}

export function Configure({ config, theme }: ConfigureProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        2. Configure
      </Text>
      <Text color={theme.foreground}>Fiber diameter: {config.fiberDiameter} µm</Text>
      <Text color={theme.foreground}>Regime: {config.regime}</Text>
      <Text color={theme.foreground}>Method: {config.method}</Text>
      <Text color={theme.foreground}>Batch size: {config.batchSize}</Text>
      <Text color={theme.muted}>Use ← → to navigate steps.</Text>
    </Box>
  );
}
