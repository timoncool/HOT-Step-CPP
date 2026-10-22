// Source checks for every release assembler, including local packaging.
import fs from 'node:fs';
import path from 'node:path';

export function checkReleaseLayout(root) {
  const errors = [];
  const read = name => fs.readFileSync(path.join(root, name), 'utf8');
  const wf = read('.github/workflows/release.yml');
  const assemblers = [
    ...['windows', 'linux', 'macos'].map(platform => {
      const match = wf.match(new RegExp(`^  build-${platform}:[\\s\\S]*?(?=^  [a-z][\\w-]*:|$(?![\\s\\S]))`, 'm'));
      return [`release.yml ${platform}`, match?.[0] ?? ''];
    }),
    ['release/build-release.ps1', read('release/build-release.ps1')],
    ['package-release.sh', read('package-release.sh')],
  ];
  for (const [name, src] of assemblers) {
    if (!src.includes('release/.env.example') && !src.includes('$ReleaseDir ".env.example"')) errors.push(`${name}: missing release env template`);
    for (const file of ['profile-path.mjs', 'README.txt', 'HOT-Step.']) {
      if (!src.includes(file)) errors.push(`${name}: missing ${file}`);
    }
    if (!/server[\\/]src[\\/]data[\\/](?:\*|\.)/.test(src)) errors.push(`${name}: resources must copy wholesale into server/data`);
    if (!/(?:\$dataOut|SRVOUT\/data|STAGING_DIR\}\/server\/data)/.test(src)) errors.push(`${name}: missing app-owned server/data target`);
    if (/(?:mkdir[^\n]*|New-Item[^\n]*)(?:\$(?:DIST|dist)|\$\{STAGING_DIR\})\/(?:models|adapters)/.test(src)) errors.push(`${name}: creates mutable folders inside app`);
    const binaryList = src.match(/\$engineFiles\s*=\s*@\(([\s\S]*?)\)/)?.[1]
      ?? [...src.matchAll(/for bin in ([^;]+); do/g)].map(m => m[1]).find(s => s.includes('neural-codec') && !s.includes('engine/build/')) ?? '';
    for (const binary of ['ace-server', 'ace-train', 'ace-midi', 'ace-caption', 'mastering', 'mp3-codec', 'neural-codec', 'vst-host', 'quantize']) {
      if (!new RegExp(`(?:^|["\\s])${binary}(?:\\.exe)?(?:$|["\\s,])`).test(binaryList)) errors.push(`${name}: missing engine binary ${binary}`);
    }
  }
  const template = read('release/.env.example');
  for (const line of template.split(/\r?\n/)) {
    if (/^(DATA_DIR|ACESTEPCPP_MODELS|ACESTEPCPP_ADAPTERS|TRAINING_DIR|LYRICS_EXPORT_DIR|MUSCRIPTOR_MODELS_DIR|WHISPER_EXE|WHISPER_MODELS_DIR|TENSORRT_LIBS)=.+/.test(line)) errors.push(`release template pins a managed path: ${line.split('=')[0]}`);
  }
  for (const launcher of ['release/HOT-Step.bat', 'release/HOT-Step.sh']) {
    const src = read(launcher);
    if (!src.includes('profile-path.mjs') || !src.includes('RESTART_MARKER')) errors.push(`${launcher}: restart must use the profile helper`);
    if (src.includes('.restart-requested') || /mkdir[^\n]*models/.test(src)) errors.push(`${launcher}: writes state inside app`);
    if (/powershell|pwsh/i.test(src)) errors.push(`${launcher}: spawns PowerShell, which behaviour-based antivirus quarantines`);
  }
  return errors;
}
