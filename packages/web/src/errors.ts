import type {ARTIDevice} from './types.js';

export type ArtiWebErrorCode =
  | 'ABORTED'
  | 'ARTIFACT_INVALID'
  | 'CONTRACT_MISMATCH'
  | 'DEVICE_UNAVAILABLE'
  | 'DISPOSED'
  | 'FETCH_FAILED'
  | 'INFERENCE_FAILED'
  | 'INITIALIZATION_FAILED'
  | 'STATE_INVALID';

export type ArtiWebErrorStage = 'fetch' | 'verify' | 'parse' | 'initialize' | 'input' | 'predict' | 'execution' | 'output' | 'state' | 'disposed';

export interface ArtiWebErrorOptions extends ErrorOptions {
  code: ArtiWebErrorCode;
  stage?: ArtiWebErrorStage;
  artifactUrl?: string | URL;
  tensorName?: string;
  expected?: unknown;
  actual?: unknown;
  device?: ARTIDevice;
}

const causes = new WeakMap<ArtiWebError, unknown>();

export function sanitizeArtifactUrl(value: string | URL): string {
  try {
    const url = value instanceof URL ? new URL(value.href) : new URL(value);
    url.username = ''; url.password = ''; url.search = ''; url.hash = '';
    return url.href;
  } catch { return '<invalid URL>'; }
}

/** Stable, machine-readable error raised by ARTI Web convenience APIs. */
export class ArtiWebError extends Error {
  readonly code: ArtiWebErrorCode;
  readonly stage?: ArtiWebErrorStage;
  readonly artifactUrl?: string;
  readonly tensorName?: string;
  readonly expected?: unknown;
  readonly actual?: unknown;
  readonly device?: ARTIDevice;
  get cause(): unknown { return causes.get(this); }

  constructor(message: string, options: ArtiWebErrorOptions) {
    super(message);
    this.name = 'ArtiWebError';
    this.code = options.code;
    this.stage = options.stage;
    this.artifactUrl = options.artifactUrl === undefined ? undefined : sanitizeArtifactUrl(options.artifactUrl);
    this.tensorName = options.tensorName;
    this.expected = options.expected;
    this.actual = options.actual;
    this.device = options.device;
    if ('cause' in options) causes.set(this, options.cause);
  }
}
