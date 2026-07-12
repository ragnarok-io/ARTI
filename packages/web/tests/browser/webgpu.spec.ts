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
  for (const name of ['half', 'fold-salience', 'fold-q', 'learned-pulse']) {
    const report = await page.evaluate((fixture) => window.runArtiParity(fixture, fixture === 'learned-pulse' ? 10 : 0, fixture === 'learned-pulse' ? 50 : 0), name);
    expect(report.device).toBe('webgpu');
    expect(Number(report.maxAbsolute)).toBeLessThanOrEqual(Number((report.tolerance as {atol: number}).atol));
    expect(Number(report.maxRelative)).toBeLessThanOrEqual(Number((report.tolerance as {rtol: number}).rtol));
    expect(Number(report.forwardIntoMaxAbsolute)).toBeLessThanOrEqual(Number((report.tolerance as {atol: number}).atol));
    reports.push({name, ...report});
  }
  console.log(JSON.stringify({adapter, reports}));
});
