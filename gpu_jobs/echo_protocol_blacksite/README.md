# Echo Protocol: Blacksite (/gpu)

Unity third-person action/story prototype assembled with GitHub Actions CPU.

- Source archive: `echo_protocol_blacksite_source.tar.gz`
- Archive SHA-256: `6be9503435854afde733e4c037c4fe2f502844d25e9dfd95bfda310f96ecbbbe`
- Visible 3D content: Quaternius **Sci-Fi Essentials Kit (Standard)**, downloaded at CI time from OpenGameArt.
- License for external pack: **CC0 1.0**
- The project itself does not contain authored FBX/OBJ/Blend models.
- GitHub Actions workflow: `.github/workflows/unity-action-story-cpu.yml`

The workflow extracts the Unity project, downloads the external models/textures on the GitHub runner, validates the mission/gameplay source, inventories the model files, and uploads a ready-to-open Unity project artifact.

Story flow: infiltrate the blacksite → retrieve archive terminal → defeat 5 ambush enemies → reach extraction.

Controls: WASD, mouse, Shift, Space, left click, R, E.
