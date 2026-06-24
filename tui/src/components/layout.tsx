import React from "react";
import { Box } from "ink";
import { Header } from "./header";
import { Sidebar, type Section } from "./sidebar";
import { Footer } from "./footer";
import type { Theme } from "../theme";

interface LayoutProps {
  section: Section;
  wizardStep: number;
  footerShortcuts: string[];
  theme: Theme;
  children: React.ReactNode;
}

export function Layout({ section, wizardStep, footerShortcuts, theme, children }: LayoutProps) {
  return (
    <Box flexDirection="column" height="100%">
      <Header theme={theme} />
      <Box flexGrow={1}>
        <Sidebar active={section} wizardStep={wizardStep} theme={theme} />
        <Box flexGrow={1} paddingX={2} paddingY={1} flexDirection="column">
          {children}
        </Box>
      </Box>
      <Footer shortcuts={footerShortcuts} theme={theme} />
    </Box>
  );
}
