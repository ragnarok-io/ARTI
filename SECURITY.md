# Security Policy

## Supported Version

Security fixes are provided for the current public 1.x release line.

## Trust Boundaries

- `.arti.st` uses SafeTensors and validates its sidecar hashes. Hashes detect
  accidental or unauthorised modification relative to the supplied lock, but
  they are not a publisher signature. Obtain artifacts from a trusted source.
- Legacy `.pt` migration and ARTI artifact loaders use PyTorch's restricted
  `weights_only=True` loader. Full pickled Python objects are not supported.
- Declarative pretrained loading rejects `trust_remote_code=True`. Review and
  instantiate models requiring remote code yourself before passing the model
  object to ARTI.
- CLI model references in `module:attribute` form intentionally import and run
  local Python factories. Treat such references and their Python environment as
  executable code, not as passive configuration.
- Downloaded models and optional dependencies remain external supply-chain
  inputs. Pin revisions and dependencies according to your deployment policy.
- ARTI rejects oversized CLI sample tensors and malformed artifact metadata,
  but callers remain responsible for bounding arbitrary model inputs, sequence
  lengths, batch sizes, and user-selected model dimensions.
- NaN and infinity values in activation configuration are rejected. NaN or
  infinity in model data follows normal PyTorch propagation semantics.

## Reporting

Report suspected vulnerabilities privately to the repository maintainers.
Do not include secrets, private model weights, or sensitive user data in a
public issue.
