import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Theme } from "../theme";

interface SelectDataProps {
  value: string;
  onChange: (path: string) => void;
  theme: Theme;
}

const RECENT_FILES = [
  "data/gfpa66_center.tif",
  "data/sample_a.tif",
  "data/sample_b.tif",
];

export function SelectData({ value, onChange, theme }: SelectDataProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  useInput((input, key) => {
    if (key.upArrow) setSelectedIdx((i) => Math.max(0, i - 1));
    if (key.downArrow) setSelectedIdx((i) => Math.min(RECENT_FILES.length - 1, i + 1));
    if (key.return) onChange(RECENT_FILES[selectedIdx]);
  });

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>
        1. Select Data
      </Text>
      <Text color={theme.muted}>Choose a recent volume or press b to browse</Text>
      {RECENT_FILES.map((file, idx) => (
        <Text key={file} color={idx === selectedIdx ? theme.highlight : theme.foreground}>
          {idx === selectedIdx ? "▸ " : "  "}
          {file}
        </Text>
      ))}
      <Box marginTop={1}>
        <Text color={theme.foreground}>Selected: {value || "none"}</Text>
      </Box>
    </Box>
  );
}
