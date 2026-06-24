import React, { useState } from "react";
import { useInput } from "ink";
import { Layout } from "./components/layout";
import { Settings } from "./components/settings";
import { useConfig } from "./hooks/useConfig";
import { loadTheme } from "./theme";
import type { Section } from "./components/sidebar";

export function App() {
  const { config, setConfig } = useConfig();
  const [section, setSection] = useState<Section>("settings");
  const [wizardStep] = useState(0);
  const theme = loadTheme(config.theme);

  useInput((input) => {
    if (input === "q") process.exit(0);
    if (input === "1") setSection("new-analysis");
    if (input === "8") setSection("settings");
  });

  return (
    <Layout
      section={section}
      wizardStep={wizardStep}
      footerShortcuts={["1 New Analysis", "8 Settings", "q Quit"]}
      theme={theme}
    >
      <Settings config={config} onChange={setConfig} theme={theme} />
    </Layout>
  );
}
