# Remote Test Bundle

This bundle is prepared for testing `server_anygrasp_vlm_test.py` on the remote
desktop.

## Contents

- `integration_case/`
  Primary end-to-end test pair with real color and 16-bit depth.
- `vlm_reference_hammer/`
  Reference VLM outputs from the current laptop-side pipeline for comparison.

## Primary Test Case

Files:

- `integration_case/color.png`
- `integration_case/depth.png`
- `integration_case/meta.json`

Suggested first test:

- Start the remote test server
- Feed `color.png` + `depth.png`
- Use the object name in `meta.json`
- Check whether the server can complete:
  OWL-v2 -> SAM -> Gemini -> mask -> SVD -> AnyGrasp

## Notes

- `depth.png` is a 16-bit depth image captured from RealSense.
- `meta.json` currently uses the same intrinsics as the current local pipeline.
- If the remote setup uses a different camera or known calibrated intrinsics,
  update `meta.json` before testing.

## Reference Files

`vlm_reference_hammer/` contains:

- `original_rgb.png`
- `target_mask.png`
- `result_overlay.png`
- `gemini_result.json`

These are useful for checking whether the remote VLM result is qualitatively
close to the current laptop-side pipeline.
