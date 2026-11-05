import assert from 'node:assert/strict';
import test from 'node:test';
import { formatDitModel, splitModel } from './modelLabels';

test('model names split into variant and quant', () => {
  const cases: [string, string, string][] = [
    ['acestep-v15-xl-base-Q8_0.gguf', 'acestep-v15-xl-base', 'Q8_0'],
    ['acestep-v15-merge-base-turbo-xl-ta-0.5-IQ4_NL.gguf', 'acestep-v15-merge-base-turbo-xl-ta-0.5', 'IQ4_NL'],
    ['acestep-v15-turbo-shift3-BF16.gguf', 'acestep-v15-turbo-shift3', 'BF16'],
    ['acestep-5Hz-lm-4B-NVFP4.gguf', 'acestep-5Hz-lm-4B', 'NVFP4'],
    // A safetensors dir joins its GGUF siblings as the unquantized option.
    ['acestep-5Hz-lm-4B', 'acestep-5Hz-lm-4B', ''],
    ['dit_fp8.onnx', 'dit_fp8.onnx', ''],
    // Bare quant tokens (backend catalogues) share the '' variant.
    ['Q4_K_M-imat', '', 'Q4_K_M-imat'],
    ['rs-4.0x0.7-q8_0', 'rs-4.0x0.7', 'q8_0'],
    ['convrot', 'convrot', ''],
  ];
  for (const [name, variant, quant] of cases) assert.deepEqual(splitModel(name), { variant, quant }, name);
});

test('variant names get the same labels as full file names', () => {
  assert.equal(formatDitModel('acestep-v15-xl-base'), 'XL Base');
  assert.equal(formatDitModel('acestep-v15-merge-base-sft-xl-ta-0.5'), 'merge-base-sft-xl-ta-0.5');
});
