# Contributing

Thank you for your interest! Bug reports, questions and ideas are welcome as
[GitHub issues](https://github.com/MohitMatrix66/epilepsy-ez-localization/issues).

## Reporting a problem

Please include:

- the command you ran;
- the log file `outputs/logs/<script>.txt`, which has the whole output and the error;
- the output of `python tools/check_install.py`.

## Changing the code

1. Make your change.
2. Run the smoke test. It must end with `SMOKE TEST PASSED`:

   ```bash
   python tools/smoke_test.py
   ```

3. Refresh the checksums. `tools/check_install.py` compares every code file with `tools/FILES.sha256`, so users notice
   an incomplete download:

   ```bash
   python tools/make_manifest.py               # update the checksums of the listed files
   python tools/make_manifest.py --add new.py  # also list a new code file
   ```

4. If the change affects the model or its results, raise `EZ_VERSION` in `config.py` (and `VERSION` in the scripts
   that check it), and describe the change and the reason in [CHANGELOG.md](CHANGELOG.md).

The GitHub Actions workflow runs `check_install.py` and the smoke test on every push.

## Ground rules

- **Never commit patient data:** no MRI, recordings or downloaded tables. `.gitignore` already excludes `outputs/`,
  `SOZ/results/`, `*.nii.gz`, `*.mgz` and `*.edf`.
- **Keep the evaluation honest:** don't choose settings by looking at the unseen patients. If you do, say so when you
  report the numbers.
- Documentation-only changes (README, `docs/`) don't need the checksums or a version change.
