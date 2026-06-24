import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface FooterProps {
  shortcuts: string[];
  theme: Theme;
}

export function Footer({ shortcuts, theme }: FooterProps) {
  return (
    <Box height={1} paddingX={1}>
      {shortcuts.map((shortcut, idx) => (
        <Text key={idx} color={theme.muted}>
          {shortcut}
          {idx < shortcuts.length - 1 ? "  " : ""}
        </Text>
      ))}
    </Box>
  );
}
