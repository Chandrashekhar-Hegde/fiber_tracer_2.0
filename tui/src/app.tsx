import React, { useState } from "react";
import { useInput, useApp } from "ink";
import { Layout } from "./components/layout";
import { WizardShell } from "./components/wizard-shell";
import { SelectData } from "./components/select-data";
import { Configure } from "./components/configure";
import { Review } from "./components/review";
import { RunWatch } from "./components/run-watch";
import { Settings } from "./components/settings";
import { useConfig } from "./hooks/useConfig";
import { useBridge } from "./hooks/useBridge";
import { loadTheme } from "./theme";
import type { Section } from "./components/sidebar";
import type { AnalysisConfig } from "./types";

const DEFAULT_CONFIG: AnalysisConfig = {
  dataPath: "",
  outputDir: "./results/run",
  voxelSpacing: [1, 1, 1],
  fiberDiameter: 10,
  regime: "auto",
  method: "otsu",
  model: "models/fiber_unet_v2_full.pt",
  batchSize: 1,
  computeMorphometry: true,
  computeOrientationTensor: true,
  computeTda: false,
};

export function App() {
  const { config, setConfig } = useConfig();
  const { exit } = useApp();
  const [section, setSection] = useState<Section>("new-analysis");
  const [wizardStep, setWizardStep] = useState(0);
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const bridge = useBridge();
  const theme = loadTheme(config.theme);

  useInput((input, key) => {
    if (input === "q") exit();
    if (input === "1") setSection("new-analysis");
    if (input === "8") setSection("settings");
    if (section === "new-analysis") {
      if (key.rightArrow || input === "n") setWizardStep((s) => Math.min(3, s + 1));
      if (key.leftArrow || input === "p") setWizardStep((s) => Math.max(0, s - 1));
      if (input === "r" && wizardStep === 2) {
        setWizardStep(3);
        bridge.run(analysisConfig);
      }
    }
  });

  const footer =
    section === "new-analysis"
      ? ["← p", "→ n", "Enter select", "r run", "q quit"]
      : ["1 Analysis", "8 Settings", "q quit"];

  return (
    <Layout section={section} wizardStep={wizardStep} footerShortcuts={footer} theme={theme}>
      {section === "new-analysis" && (
        <WizardShell step={wizardStep} theme={theme}>
          {wizardStep === 0 && (
            <SelectData
              value={analysisConfig.dataPath}
              onChange={(path) => setAnalysisConfig((c) => ({ ...c, dataPath: path }))}
              theme={theme}
            />
          )}
          {wizardStep === 1 && (
            <Configure config={analysisConfig} onChange={setAnalysisConfig} theme={theme} />
          )}
          {wizardStep === 2 && <Review config={analysisConfig} theme={theme} />}
          {wizardStep === 3 && (
            <RunWatch status={bridge.status} progress={bridge.progress} logs={bridge.logs} theme={theme} />
          )}
        </WizardShell>
      )}
      {section === "settings" && <Settings config={config} onChange={setConfig} theme={theme} />}
    </Layout>
  );
}
