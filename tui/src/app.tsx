import React, { useRef, useState, useCallback } from "react";
import { useInput, useApp } from "ink";
import { Layout } from "./components/layout";
import { WizardShell } from "./components/wizard-shell";
import { SelectData } from "./components/select-data";
import { Configure } from "./components/configure";
import { Review } from "./components/review";
import { RunWatch } from "./components/run-watch";
import { Dashboard } from "./components/dashboard";
import { History } from "./components/history";
import { Experiments } from "./components/experiments";
import { ModelRegistry } from "./components/model-registry";
import { Training, TrainingRef } from "./components/training";
import { Logs } from "./components/logs";
import { Settings } from "./components/settings";
import { useConfig } from "./hooks/useConfig";
import { useBridge } from "./hooks/useBridge";
import { useHistory } from "./hooks/useHistory";
import { loadTheme } from "./theme";
import type { Section } from "./components/sidebar";
import type { AnalysisConfig, Model, Experiment, TrainingOptions } from "./types";

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
  const history = useHistory();
  const theme = loadTheme(config.theme);
  const trainingRef = useRef<TrainingRef>(null);

  const onInput = useCallback(
    (input: string, key: { rightArrow: boolean; leftArrow: boolean }) => {
      if (input === "q") exit();
      if (input === "1") setSection("new-analysis");
      if (input === "2") setSection("dashboard");
      if (input === "3") setSection("history");
      if (input === "4") setSection("experiments");
      if (input === "5") setSection("model-registry");
      if (input === "6") setSection("training");
      if (input === "7") setSection("logs");
      if (input === "8") setSection("settings");
      if (section === "new-analysis") {
        if (key.rightArrow || input === "n") setWizardStep((s) => Math.min(3, s + 1));
        if (key.leftArrow || input === "p") setWizardStep((s) => Math.max(0, s - 1));
        if (input === "r" && wizardStep === 2) {
          setWizardStep(3);
          bridge.run(analysisConfig);
        }
      }
      if (section === "training" && input === "s") {
        trainingRef.current?.start();
      }
    },
    [exit, section, wizardStep, bridge, analysisConfig]
  );

  useInput(onInput);

  const footer =
    section === "new-analysis"
      ? ["← p", "→ n", "Enter select", "r run", "q quit"]
      : ["1 Analysis", "2 Dashboard", "3 History", "4 Experiments", "5 Registry", "6 Training", "7 Logs", "8 Settings", "q quit"];

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
            <Configure config={analysisConfig} theme={theme} />
          )}
          {wizardStep === 2 && <Review config={analysisConfig} theme={theme} />}
          {wizardStep === 3 && (
            <RunWatch status={bridge.status} progress={bridge.progress} logs={bridge.logs} error={bridge.error} theme={theme} />
          )}
        </WizardShell>
      )}
      {section === "dashboard" && <Dashboard summary={undefined} theme={theme} />}
      {section === "history" && <History history={history.history} theme={theme} />}
      {section === "experiments" && <Experiments theme={theme} />}
      {section === "model-registry" && <ModelRegistry theme={theme} />}
      {section === "training" && <Training ref={trainingRef} theme={theme} />}
      {section === "logs" && <Logs logs={bridge.logs} theme={theme} />}
      {section === "settings" && <Settings config={config} onChange={setConfig} theme={theme} />}
    </Layout>
  );
}
