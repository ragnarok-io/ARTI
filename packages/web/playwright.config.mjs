import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 60_000,
  workers: 1,
  use: {
    browserName: 'chromium',
    headless: true,
    launchOptions: {
      executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
    },
  },
  webServer: {
    command: 'node node_modules/vite/bin/vite.js --config vite.browser.config.mjs',
    port: 4178,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
