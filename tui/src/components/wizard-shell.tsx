import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

interface WizardShellProps {
  step: number;
  theme: Theme;
  children: React.ReactNode;
}

const STEPS = ["Select Data", "Configure", "Review", "Run & Watch"];

export function WizardShell({ step, theme, children }: WizardShellProps) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Box marginBottom={1}>
        {STEPS.map((label, idx) => (
          <Text key={label} color={idx === step ? theme.highlight : theme.muted}>
            {idx === step ? ` ${idx + 1}. ${label} ` : ` ${idx + 1}. ${label} `}
          </Text>
        ))}
      </Box>
      {children}
    </Box>
  );
}
