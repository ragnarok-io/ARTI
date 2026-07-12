import {beforeEach, describe, expect, it, vi} from 'vitest';

const ortMock = vi.hoisted(() => {
  class Tensor {
    static fromGpuBuffer = vi.fn();
    readonly location: string;
    readonly type = 'float32';
    readonly dims: readonly number[];
    readonly dispose = vi.fn();
    constructor(_type: string, _data: Float32Array, dims: readonly number[], location = 'cpu') {
      this.dims = dims;
      this.location = location;
    }
    async getData(): Promise<Float32Array> { return new Float32Array(this.dims.reduce((a, b) => a * b, 1)); }
  }
  return {Tensor, env: {webgpu: {device: Promise.resolve(undefined)}}};
});

vi.mock('onnxruntime-web/webgpu', () => ({
  Tensor: ortMock.Tensor,
  env: ortMock.env,
  InferenceSession: {create: vi.fn()},
}));

import {ARTIWebModule} from '../src/runtime.js';
import {ARTIStatefulWebModule} from '../src/stateful.js';
import type {ARTIStatefulWebManifest, ARTIWebManifest} from '../src/generated/contract.js';
import type {TensorMap} from '../src/types.js';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

const tensorContract = (name: string, shape: Array<number | string>) => ({name, dtype: 'float32' as const, shape});
const input = () => new ortMock.Tensor('float32', new Float32Array(1), [1]) as never;

describe('runtime lifecycle', () => {
  beforeEach(() => vi.clearAllMocks());

  it('rejects new stateless runs while dispose waits for an in-flight run', async () => {
    const execution = deferred<TensorMap>();
    const session = {run: vi.fn(() => execution.promise), release: vi.fn(async () => undefined)};
    const manifest = {inputs: [tensorContract('x', [1])], outputs: [tensorContract('y', [1])]} as unknown as ARTIWebManifest;
    const module = new ARTIWebModule(session as never, manifest, 'wasm');

    const running = module.run({x: input()});
    const disposing = module.dispose();
    await expect(module.run({x: input()})).rejects.toThrow(/disposed/);
    expect(session.release).not.toHaveBeenCalled();

    const output = new ortMock.Tensor('float32', new Float32Array(1), [1]) as never;
    execution.resolve({y: output});
    await expect(running).resolves.toEqual({y: output});
    await disposing;
    expect(session.release).toHaveBeenCalledOnce();
    await module.dispose();
    expect(session.release).toHaveBeenCalledOnce();
  });

  it('serializes stateful operations and queues dispose behind accepted work', async () => {
    const first = deferred<TensorMap>();
    const events: string[] = [];
    vi.spyOn(ortMock.Tensor.prototype, 'getData').mockImplementation(async function () {
      events.push('snapshot');
      return new Float32Array(this.dims.reduce((a, b) => a * b, 1));
    });
    const output = new ortMock.Tensor('float32', new Float32Array(1), [1]) as never;
    const session = {
      run: vi.fn(async () => { events.push('run'); return first.promise; }),
      release: vi.fn(async () => { events.push('release'); }),
    };
    const module = statefulModule(session, 'wasm');

    const running = module.run('read', {x: input()});
    const snapshot = module.snapshot();
    const disposing = module.dispose();
    await expect(module.run('read', {x: input()})).rejects.toThrow(/disposed/);
    expect(events).toEqual(['run']);

    first.resolve({y: output});
    await running;
    await snapshot;
    await disposing;
    expect(events).toEqual(['run', 'snapshot', 'release']);
  });

  it('serializes concurrent commits so the second observes the first adopted state', async () => {
    const first = deferred<TensorMap>();
    const second = deferred<TensorMap>();
    const session = {run: vi.fn().mockImplementationOnce(() => first.promise).mockImplementationOnce(() => second.promise), release: vi.fn(async () => undefined)};
    const module = committingModule(session);
    const one = module.commit('update', {x: input()});
    const two = module.commit('update', {x: input()});
    await vi.waitFor(() => expect(session.run).toHaveBeenCalledTimes(1));
    const firstState = input();
    first.resolve({next_state: firstState});
    await one;
    await vi.waitFor(() => expect(session.run).toHaveBeenCalledTimes(2));
    expect((session.run.mock.calls[1]![0] as TensorMap).state).toBe(firstState);
    second.resolve({next_state: input()});
    await two;
    await module.dispose();
  });

  it('rejects state aliases without disposing the live previous state', async () => {
    const session = {run: vi.fn(), release: vi.fn(async () => undefined)};
    const module = committingModule(session);
    const initial = input();
    session.run.mockResolvedValueOnce({next_state: initial});
    await module.commit('update', {x: input()});

    session.run.mockImplementationOnce(async (feeds: TensorMap) => ({next_state: feeds.state!, diagnostic: feeds.state!}));
    await expect(module.commit('update', {x: input()})).rejects.toThrow(/aliases new state.*previous state/);
    expect((initial as unknown as InstanceType<typeof ortMock.Tensor>).dispose).not.toHaveBeenCalled();
    await module.dispose();
    expect((initial as unknown as InstanceType<typeof ortMock.Tensor>).dispose).toHaveBeenCalledOnce();
  });

  it('rejects aliases between adopted state and ordinary outputs', async () => {
    const aliased = input();
    const session = {run: vi.fn(async () => ({next_state: aliased, diagnostic: aliased})), release: vi.fn(async () => undefined)};
    const module = committingModule(session);
    await expect(module.commit('update', {x: input()})).rejects.toThrow(/aliases output diagnostic with state state/);
    expect((aliased as unknown as InstanceType<typeof ortMock.Tensor>).dispose).toHaveBeenCalledOnce();
    await module.dispose();
  });

  it('rejects aliases between newly adopted state tensors', async () => {
    const aliased = input();
    const state = [tensorContract('left', ['batch']), tensorContract('right', ['batch'])];
    const manifest = {
      state, limits: {max_state_bytes_per_batch: 8},
      entrypoints: {update: {
        file: 'update.onnx', inputs: [tensorContract('x', ['batch']), ...state],
        outputs: [tensorContract('next_left', ['batch']), tensorContract('next_right', ['batch'])],
        state_outputs: {left: 'next_left', right: 'next_right'},
      }},
    } as unknown as ARTIStatefulWebManifest;
    const session = {run: vi.fn(async () => ({next_left: aliased, next_right: aliased})), release: vi.fn(async () => undefined)};
    const module = new ARTIStatefulWebModule(new Map([['update', session as never]]), manifest, 'wasm', 1024, async () => { throw new Error('unused'); });
    await expect(module.commit('update', {x: input()})).rejects.toThrow(/aliases new state left and right/);
    expect((aliased as unknown as InstanceType<typeof ortMock.Tensor>).dispose).toHaveBeenCalledOnce();
    await module.dispose();
  });

  it('makes reset await the operations accepted before it', async () => {
    const execution = deferred<TensorMap>();
    const session = {run: vi.fn(() => execution.promise), release: vi.fn(async () => undefined)};
    const module = statefulModule(session, 'wasm');
    const running = module.run('read', {x: input()});
    const resetting = module.reset();
    let resetFinished = false;
    void resetting.then(() => { resetFinished = true; });
    await Promise.resolve();
    expect(resetFinished).toBe(false);
    execution.resolve({y: input()});
    await running;
    await resetting;
    expect(module.stateInfo().batchSize).toBe(0);
    await module.dispose();
  });

  it('rolls back failed GPU state creation, disposes wrappers and buffers, then retries', async () => {
    const destroyed: Array<ReturnType<typeof vi.fn>> = [];
    const device = {
      createBuffer: vi.fn(() => {
        const destroy = vi.fn();
        destroyed.push(destroy);
        return {getMappedRange: () => new ArrayBuffer(16), unmap: vi.fn(), destroy};
      }),
    };
    ortMock.env.webgpu.device = Promise.resolve(device as never);
    (globalThis as {GPUBufferUsage?: unknown}).GPUBufferUsage = {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4};
    const wrappers: InstanceType<typeof ortMock.Tensor>[] = [];
    ortMock.Tensor.fromGpuBuffer
      .mockImplementationOnce((_buffer, options) => {
        const wrapper = new ortMock.Tensor('float32', new Float32Array(1), options.dims, 'gpu-buffer');
        wrappers.push(wrapper);
        return wrapper;
      })
      .mockImplementationOnce(() => { throw new Error('wrapper failed'); })
      .mockImplementation((_buffer, options) => {
        const wrapper = new ortMock.Tensor('float32', new Float32Array(1), options.dims, 'gpu-buffer');
        wrappers.push(wrapper);
        return wrapper;
      });
    const session = {run: vi.fn(async () => ({y: input()})), release: vi.fn(async () => undefined)};
    const module = statefulModule(session, 'webgpu', [tensorContract('a', ['batch']), tensorContract('b', ['batch'])]);

    await expect(module.run('read', {x: input()})).rejects.toThrow('wrapper failed');
    expect(wrappers[0]!.dispose).toHaveBeenCalledOnce();
    expect(destroyed.slice(0, 2).every((destroy) => destroy.mock.calls.length === 1)).toBe(true);
    expect(module.stateInfo().batchSize).toBe(0);

    await expect(module.run('read', {x: input()})).resolves.toBeDefined();
    expect(device.createBuffer).toHaveBeenCalledTimes(4);
    await module.dispose();
    expect(wrappers.slice(1).every((wrapper) => wrapper.dispose.mock.calls.length === 1)).toBe(true);
    expect(destroyed.every((destroy) => destroy.mock.calls.length === 1)).toBe(true);
  });
});

function statefulModule(session: object, device: 'wasm' | 'webgpu', state = [tensorContract('state', ['batch'])]): ARTIStatefulWebModule {
  const manifest = {
    state,
    limits: {max_state_bytes_per_batch: state.length * 4},
    entrypoints: {
      read: {
        file: 'read.onnx',
        inputs: [tensorContract('x', ['batch']), ...state],
        outputs: [tensorContract('y', ['batch'])],
      },
    },
  } as unknown as ARTIStatefulWebManifest;
  return new ARTIStatefulWebModule(new Map([['read', session as never]]), manifest, device, 1024, async () => { throw new Error('unused'); });
}

function committingModule(session: object): ARTIStatefulWebModule {
  const state = tensorContract('state', ['batch']);
  const manifest = {
    state: [state], limits: {max_state_bytes_per_batch: 4},
    entrypoints: {update: {file: 'update.onnx', inputs: [tensorContract('x', ['batch']), state], outputs: [tensorContract('next_state', ['batch'])], state_outputs: {state: 'next_state'}}},
  } as unknown as ARTIStatefulWebManifest;
  return new ARTIStatefulWebModule(new Map([['update', session as never]]), manifest, 'wasm', 1024, async () => { throw new Error('unused'); });
}
