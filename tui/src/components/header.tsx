import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface HeaderProps {
  theme: Theme;
}

export function Header({ theme }: HeaderProps) {
  return (
    <Box height={1} justifyContent="space-between" paddingX={1}>
      <Text color={theme.accent}>Fiber Tracer TUI</Text>
      <Text color={theme.muted}>v3.2.0 | ? help | q quit</Text>
    </Box>
  );
}
