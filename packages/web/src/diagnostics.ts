import type {ActiveARTIDevice} from './generated/contract.js';

export type LoadStage = 'manifest' | 'lock' | 'model' | 'verify' | 'initialize' | 'ready';

export interface LoadProgress {
  stage: LoadStage;
  loadedBytes?: number;
  totalBytes?: number;
  device?: ActiveARTIDevice;
}

export type LoadProgressCallback = (progress: Readonly<LoadProgress>) => void;

export interface LoadAttemptDiagnostic {
  device: ActiveARTIDevice;
  startedAt: number;
  finishedAt: number;
  success: boolean;
  error?: {name: string; message: string; code?: string};
}

export function summarizeDiagnosticError(error: unknown): {name: string; message: string; code?: string} {
  if (error instanceof Error) {
    const code = 'code' in error && typeof error.code === 'string' ? error.code : undefined;
    return {...(code === undefined ? {} : {code}), name: error.name, message: 'Initialization attempt failed'};
  }
  return {name: 'Error', message: 'Initialization attempt failed'};
}

export interface LoadDiagnostics {
  artifactUrl: string;
  startedAt: number;
  finishedAt: number;
  attempts: readonly LoadAttemptDiagnostic[];
  selectedDevice?: ActiveARTIDevice;
}
