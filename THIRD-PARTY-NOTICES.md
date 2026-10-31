# Third-party notices

This repository is licensed **GPL-3.0-or-later** (see [LICENSE](LICENSE)). It contains portions of one
third-party work, which is licensed under the MIT licence. GPL-3.0-or-later can incorporate MIT-licensed code
provided the MIT copyright and permission notice is kept, and this file is where it is kept. It is also named in
the header of each file concerned.

## bullet

- Project: <https://github.com/jw1912/bullet> - the NNUE trainer these examples are compiled inside
- Copyright (c) 2023 Jamie Whiting
- Licence: MIT
- Pinned commit: `da426a49b2a4b79a55a87031cd51c89bdf1ff030` (the one `bullet/setup_bullet.sh` checks out)
- Taken from: `examples/simple.rs` at that commit (git blob `c28ab5d04e1d4da7db91fd881ec23ca03bce5c0e`)

| File here | What derives from bullet's `examples/simple.rs` |
|---|---|
| `bullet/pawnstar.rs` | The skeleton of the trainer: the `use bullet_lib::{...}` import block, the `QA`/`QB`/`SCALE` constants, the `ValueTrainerBuilder` chain, the `SavedFormat` quantisation list, the loss function, the SCReLU forward pass, and the `TrainingSchedule` layout. |
| `bullet/pawnstar_eval.rs` | The reference inference, which follows the example inference at the foot of `simple.rs`: the SCReLU function, the `Net` field names and the order of the output dequantisation. The rest of the file is not bullet's. |

Everything else in this repository was written for it or comes from `pawnstar` and `zenith`, recorded file by
file in the maintainer's provenance log; none of it carries bullet's code. bullet itself is **not** vendored here:
`bullet/setup_bullet.sh` clones it at the pinned commit, so it is a dependency, under its own licence.
Reading bullet's on-disk record format in `pytorch/bulletformat.py` is compatibility with a file format and
carries no bullet code.

### The MIT licence text, as published at that commit

```text
MIT License

Copyright (c) 2023 Jamie Whiting

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
