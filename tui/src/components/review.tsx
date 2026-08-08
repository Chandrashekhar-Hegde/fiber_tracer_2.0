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
      <Text color={theme.foreground}>Threshold: {config.thresholdMethod}</Text>
      <Text color={theme.foreground}>
        Tracking: {config.computeTracking ? "on" : "off"}
      </Text>
      <Text color={theme.foreground}>Model: {config.model}</Text>
      <Text color={theme.foreground}>
        DVC: {config.computeDvc ? "on" : "off"}
        {config.computeDvc
          ? ` (reference: ${config.dvcReferencePath || "none"}, deformed: ${config.dvcDeformedPath || "none"})`
          : ""}
      </Text>
      <Text color={theme.foreground}>
        DIC: {config.computeDic ? "on" : "off"}
        {config.computeDic
          ? ` (reference: ${config.dicReferencePath || "none"}, deformed: ${config.dicDeformedPath || "none"})`
          : ""}
      </Text>
      <Text color={theme.foreground}>Digital twin: {config.computeTwin ? "on" : "off"}</Text>
      <Text color={theme.success}>Press r to run</Text>
    </Box>
  );
}
