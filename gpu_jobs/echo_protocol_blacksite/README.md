# Echo Protocol: Blacksite (/gpu)

Third-person Unity action/story prototype assembled and validated on GitHub Actions CPU.

External 3D content is downloaded, not generated:
- ECHO-7 player: Quaternius Animated Characters Pack (CC0)
- Enemies, weapons and props: Quaternius Sci-Fi Essentials Kit Standard (CC0)

Story: infiltrate the blacksite, retrieve the encrypted archive, defeat the five-unit security ambush, and reach extraction.

Controls: WASD, mouse, Left Shift, Space, left click, R, E, Esc.

The GitHub workflow extracts the Unity 2022.3 LTS source, applies the humanoid-player patch, downloads both external packs, validates mission/gameplay code and the no-authored-model constraint, inventories models, then uploads a ready-to-open Unity project artifact.
