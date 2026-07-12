import {describe, expect, it, vi} from 'vitest';

import {ArtiWebError} from '../src/errors.js';
import {download} from '../src/runtime.js';

describe('bounded artifact downloads', () => {
  it('rejects an oversized Content-Length before reading the body', async () => {
    const cancel = vi.fn();
    const response = new Response(new ReadableStream({cancel}), {headers: {'content-length': '20'}});
    await expect(download(new URL('https://arti.test/model.onnx'), 'model.onnx', 10, async () => response, undefined, undefined, 'model')).rejects.toMatchObject({code: 'ARTIFACT_INVALID', actual: 20, expected: 10});
    expect(cancel).not.toHaveBeenCalled();
  });

  it('enforces the limit while streaming and reports progress', async () => {
    const updates: number[] = [];
    const response = new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array(6)); controller.enqueue(new Uint8Array(6)); controller.close(); },
    }));
    await expect(download(new URL('https://arti.test/model.onnx'), 'model.onnx', 10, async () => response, undefined, (event) => updates.push(event.loadedBytes ?? 0), 'model')).rejects.toBeInstanceOf(ArtiWebError);
    expect(updates).toEqual([6]);
  });

  it('requires Content-Length to equal expectedSize', async () => {
    const response = new Response(new Uint8Array(4), {headers: {'content-length': '4'}});
    await expect(download(new URL('https://user:pass@arti.test/model.onnx?token=secret#x'), 'model.onnx', 10, async () => response, undefined, undefined, 'model', 3)).rejects.toMatchObject({expected: 3, actual: 4, artifactUrl: 'https://arti.test/model.onnx'});
  });

  it('uses expectedSize as the streaming limit when it is smaller', async () => {
    const response = new Response(new ReadableStream({start(controller) { controller.enqueue(new Uint8Array(4)); controller.close(); }}));
    await expect(download(new URL('https://arti.test/model.onnx'), 'model.onnx', 10, async () => response, undefined, undefined, 'model', 3)).rejects.toMatchObject({expected: 3, actual: 4});
  });

  it('passes AbortSignal to fetch and returns a structured abort', async () => {
    const controller = new AbortController();
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.signal).toBe(controller.signal);
      controller.abort('stop');
      throw new DOMException('aborted', 'AbortError');
    });
    await expect(download(new URL('https://arti.test/model.onnx'), 'model.onnx', 10, fetcher, controller.signal, undefined, 'model')).rejects.toMatchObject({code: 'ABORTED'});
  });
});
