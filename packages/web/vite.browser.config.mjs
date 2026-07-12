import { defineConfig } from 'vite';
import { createReadStream } from 'node:fs';
import path from 'node:path';

const wasmFile = path.resolve('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm');

export default defineConfig({
  root: '.',
  publicDir: path.resolve('../../.tmp/web-fixtures'),
  plugins: [{
    name: 'arti-ort-wasm',
    configureServer(server) {
      server.middlewares.use('/ort-runtime.wasm', (_request, response) => {
        response.setHeader('Content-Type', 'application/wasm');
        createReadStream(wasmFile).pipe(response);
      });
    },
  }],
  server: {host: '127.0.0.1', port: 4178, strictPort: true},
});
