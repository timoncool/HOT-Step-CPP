// trainingPlugins.tsx — the registry of training plug-ins, keyed by target.
//
// One entry per backend; TrainingStudio, PhaseStepper, DatasetAssetChips and
// TrainPanel read everything target-specific from here instead of branching on
// backend ids themselves.

import type { TrainingPlugin, TrainingTargetId } from './trainingPluginTypes';
import { acePlugin } from './plugins/ace';
import { mm3Plugin } from './plugins/mm3';
import { sa3Plugin } from './plugins/sa3';
import { yue2Plugin } from './plugins/yue2';

export const TRAINING_PLUGINS: Record<TrainingTargetId, TrainingPlugin> = {
  ace: acePlugin,
  yue2: yue2Plugin,
  'minimax-m3': mm3Plugin,
  sa3: sa3Plugin,
};

export const getTrainingPlugin = (target: TrainingTargetId): TrainingPlugin => TRAINING_PLUGINS[target];
