# Changelog

## 3.1.0a2

- Make tagged releases wait for the same package and documentation CI used by
  branch builds before package publication.
- Align release readiness checks and the release guide with the public package
  layout, removing requirements for private benchmark reports.

This release hardens the release pipeline; the public API remains alpha. See
`STABILITY.md` for compatibility and claim boundaries.

## 3.1.0a1

- Rebased the public package on the current program-runtime API.
- Promoted the current `ARTILayer` host, Formula Fabric, federated programs,
  resource graph, and execution interfaces into the published alpha surface.
- Moved retired layered Recall implementations under `arti.legacy` and
  refreshed public documentation, examples, schemas, and tests.

This release is an alpha API transition. See `STABILITY.md` for compatibility
and claim boundaries.
