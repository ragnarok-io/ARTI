import { expect, test } from '@playwright/test';

test('uses a hardware WebGPU adapter and matches PyTorch', async ({page}) => {
  page.on('console', (message) => console.log(`[browser:${message.type()}] ${message.text()}`));
  const probe = await page.request.get('http://127.0.0.1:4178/browser/');
  if (!probe.ok()) throw new Error(`Vite returned ${probe.status()}: ${await probe.text()}`);
  await page.goto('http://127.0.0.1:4178/browser/');
  const adapter = await page.evaluate(() => window.artiWebGPUInfo());
  expect(adapter.isFallbackAdapter).toBe(false);
  expect([adapter.vendor, adapter.architecture, adapter.device, adapter.description].some((value) => Boolean(value))).toBe(true);

  const reports: Record<string, unknown>[] = [];
  for (const name of ['half', 'fold-salience', 'fold-q', 'learned-pulse', 'generic-affine']) {
    const report = await page.evaluate((fixture) => window.runArtiParity(fixture, fixture === 'learned-pulse' ? 10 : 0, fixture === 'learned-pulse' ? 50 : 0), name);
    expect(report.device).toBe('webgpu');
    expect(Number(report.maxAbsolute)).toBeLessThanOrEqual(Number((report.tolerance as {atol: number}).atol));
    expect(Number(report.maxRelative)).toBeLessThanOrEqual(Number((report.tolerance as {rtol: number}).rtol));
    expect(Number(report.forwardIntoMaxAbsolute)).toBeLessThanOrEqual(Number((report.tolerance as {atol: number}).atol));
    reports.push({name, ...report});
  }
  console.log(JSON.stringify({adapter, reports}));
});

test('keeps committed Recall state on WebGPU and suppresses unseen reads', async ({page}) => {
  await page.goto('http://127.0.0.1:4178/browser/');
  const report = await page.evaluate(() => window.runStatefulRecall());
  expect(report.device).toBe('webgpu');
  expect(Object.values((report.state as {locations: Record<string, string>}).locations)).toEqual(['gpu-buffer', 'gpu-buffer', 'gpu-buffer']);
  expect(Number((report.state as {bytes: number}).bytes)).toBe(Number(report.firstCommittedBytes));
  expect(Number(report.seenRecognition)).toBeGreaterThan(0.8);
  expect(Number(report.unseenRecognition)).toBeLessThan(0.05);
  console.log(JSON.stringify(report));
});

test('executes Python-owned FusionPulse inspect outputs on WebGPU', async ({page}) => {
  await page.goto('http://127.0.0.1:4178/browser/');
  const report = await page.evaluate(() => window.runFusionPulseInspect());
  expect(report.device).toBe('webgpu');
  expect(Number(report.maxAbsolute)).toBeLessThanOrEqual(1e-4);
  expect(Number(report.maxRelative)).toBeLessThanOrEqual(1e-3);
  expect(report.survivalChanged).toBe(true);
  expect(report.sourceMappingChanged).toBe(true);
  expect(report.roles).toMatchObject({fused: 'primary', workspace: 'workspace', survival: 'diagnostic'});
  console.log(JSON.stringify(report));
});

test('runs a Python-exported artifact inside a module Worker with transferable tensors', async ({page}) => {
  await page.goto('http://127.0.0.1:4178/browser/');
  const report = await page.evaluate(() => window.runWorkerSmoke());
  expect(report.device).toBe('webgpu');
  expect(report.inputBuffersDetached).toBe(true);
  expect(Number(report.maxAbsolute)).toBeLessThanOrEqual(1e-5);
  console.log(JSON.stringify(report));
});

test('inspects Python-owned workspaces inside a module Worker', async ({page}) => {
  await page.goto('http://127.0.0.1:4178/browser/');
  const report = await page.evaluate(() => window.runWorkerInspect());
  expect(report.device).toBe('webgpu');
  expect(report.outputNames).toEqual(['fused', 'survival', 'pulse_mask', 'workspace', 'unfold_source_index']);
  expect(report.roles).toMatchObject({fused: 'primary', survival: 'diagnostic', workspace: 'workspace'});
  expect(Number(report.maxAbsolute)).toBeLessThanOrEqual(1e-4);
  expect(report.inputBuffersDetached).toBe(true);
  console.log(JSON.stringify(report));
});
