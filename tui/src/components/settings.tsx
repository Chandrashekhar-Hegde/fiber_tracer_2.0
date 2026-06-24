import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { UserConfig } from "../types";

interface SettingsProps {
  config: UserConfig;
  onChange: (config: UserConfig) => void;
  theme: Theme;
}

export function Settings({ config, onChange, theme }: SettingsProps) {
  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        Settings
      </Text>
      <Text color={theme.foreground}>Theme: {config.theme}</Text>
      <Text color={theme.muted}>Default output: {config.defaultOutputDir}</Text>
      <Text color={theme.muted}>Default model: {config.defaultModel}</Text>
    </Box>
  );
}
