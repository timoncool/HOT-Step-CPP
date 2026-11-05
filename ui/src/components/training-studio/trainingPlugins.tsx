// trainingPlugins.tsx — the registry of training plug-ins, keyed by target.
//
// One entry per backend; TrainingStudio, PhaseStepper, DatasetAssetChips and
// TrainPanel read everything target-specific from here instead of branching on
// backend ids themselves.

import type { TrainingPlugin, TrainingTargetId } from './trainingPluginTypes';
import { acePlugin } from './plugins/ace';
import { mm3Plugin } from './plugins/mm3';
import { sa3Plugin } from './plugins/sa3';
import { createHeartmulaPlugin } from './plugins/heartmula';
import { heartmulaChips, heartmulaReadiness } from './plugins/chips';
import { TRAINING_TARGETS } from './trainingTargets';
import { useHeartmulaController } from './useHeartmulaController';
import { yue2Plugin } from './plugins/yue2';

const heartmulaPlugin = createHeartmulaPlugin({
  target: { ...TRAINING_TARGETS.heartmula, chips: heartmulaChips, readiness: heartmulaReadiness },
  useController: useHeartmulaController,
});

export const TRAINING_PLUGINS: Record<TrainingTargetId, TrainingPlugin> = {
  ace: acePlugin,
  yue2: yue2Plugin,
  'minimax-m3': mm3Plugin,
  sa3: sa3Plugin,
  heartmula: heartmulaPlugin,
};

export const getTrainingPlugin = (target: TrainingTargetId): TrainingPlugin => TRAINING_PLUGINS[target];
