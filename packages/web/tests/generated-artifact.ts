// Generated from the contract fixture in tests/test_web_export.py. Do not edit by hand.
import type { ARTIWebModule, Tensor } from '@arti-fit/web';

export const descriptor = {"format":"arti.web","format_version":2,"inputs":["signal","gate"],"outputs":["result","salience"]} as const;
export type ArtifactInputs = { "signal": Tensor; "gate": Tensor };
export type ArtifactOutputs = { "result": Tensor; "salience": Tensor };

export class ArtifactClient {
  constructor(readonly module: ARTIWebModule) {}

  async run(inputs: ArtifactInputs): Promise<ArtifactOutputs> {
    return await this.module.run(inputs) as ArtifactOutputs;
  }
}
