import React from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";

export type Section =
  | "new-analysis"
  | "dashboard"
  | "history"
  | "experiments"
  | "model-registry"
  | "training"
  | "logs"
  | "settings";

interface SidebarProps {
  active: Section;
  wizardStep: number;
  theme: Theme;
}

const ITEMS: { key: Section; label: string; shortcut: string }[] = [
  { key: "new-analysis", label: "New Analysis", shortcut: "1" },
  { key: "dashboard", label: "Dashboard", shortcut: "2" },
  { key: "history", label: "History", shortcut: "3" },
  { key: "experiments", label: "Experiments", shortcut: "4" },
  { key: "model-registry", label: "Model Registry", shortcut: "5" },
  { key: "training", label: "Training", shortcut: "6" },
  { key: "logs", label: "Logs", shortcut: "7" },
  { key: "settings", label: "Settings", shortcut: "8" },
];

export function Sidebar({ active, wizardStep, theme }: SidebarProps) {
  return (
    <Box width={24} flexDirection="column" borderStyle="single" borderColor={theme.border} paddingX={1}>
      <Text bold color={theme.accent}>
        Fiber Tracer
      </Text>
      {ITEMS.map((item) => {
        const isActive = active === item.key;
        const isWizard = item.key === "new-analysis";
        return (
          <Box key={item.key} flexDirection="column" marginTop={isWizard ? 1 : 0}>
            <Text color={isActive ? theme.highlight : theme.foreground}>
              {isActive ? "▸ " : "  "}
              {item.label}
            </Text>
            {isWizard && isActive && (
              <Box flexDirection="column" paddingLeft={2}>
                {["1. Select Data", "2. Configure", "3. Review", "4. Run & Watch"].map((step, idx) => (
                  <Text key={step} color={idx === wizardStep ? theme.highlight : theme.muted}>
                    {step}
                  </Text>
                ))}
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
