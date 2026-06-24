import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { AnalysisConfig } from "../types";

interface ReviewProps {
  config: AnalysisConfig;
  theme: Theme;
}

export function Review({ config, theme }: ReviewProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        3. Review
      </Text>
      <Text color={theme.foreground}>Data: {config.dataPath}</Text>
      <Text color={theme.foreground}>Output: {config.outputDir}</Text>
      <Text color={theme.foreground}>Method: {config.method}</Text>
      <Text color={theme.foreground}>Model: {config.model}</Text>
      <Text color={theme.success}>Press r to run</Text>
    </Box>
  );
}
