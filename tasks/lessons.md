# Lessons Learned

## Deployment

### L1: HF Spaces always needs yaml header in README.md
- **Mistake**: Pushed code to HF without yaml header — spaces showed default template
- **Fix**: Always prepend `---\nsdk: docker\n---` to README before pushing to HF
- **Rule**: Never push to HF without verifying README has valid yaml frontmatter

### L2: HF Spaces ignores `dockerfile:` metadata key
- **Mistake**: Assumed `dockerfile: Dockerfile.streamlit` in yaml header would work
- **Fix**: Root `Dockerfile` must contain the correct build instructions for that space
- **Rule**: For Streamlit space, swap root `Dockerfile` content with `Dockerfile.streamlit` before pushing

### L3: Shallow clone breaks HF push
- **Mistake**: Used `fetch-depth: 1` in GitHub Actions — HF rejected push with "shallow update not allowed"
- **Fix**: Always use `fetch-depth: 0` (full clone) for deploy jobs
- **Rule**: Deploy jobs must use full clone, test/build jobs can use shallow

### L4: Trailing spaces in environment variables break URLs
- **Mistake**: `API_URL` was set with two trailing spaces → `%20%20` appended to hostname → DNS failure
- **Fix**: Delete and retype the value carefully with no trailing spaces
- **Rule**: Always show the actual URL in error messages so trailing space issues are immediately visible

### L5: Python heredoc inside GitHub Actions YAML breaks parsing
- **Mistake**: Used `python3 - <<'EOF'` with yaml-like content inside `run:` block — GitHub YAML parser read `title:`, `sdk:` etc. as workflow keys
- **Fix**: Use `printf '%s\n' ...` or Python one-liner strings instead of heredoc
- **Rule**: Never use heredoc with yaml-like content inside a GitHub Actions workflow file

## CI/CD

### L6: scikit-learn version must match between training and serving
- **Mistake**: Models saved with scikit-learn 1.8.0 but requirements.txt had 1.4.2 → import errors
- **Fix**: Pin exact versions in requirements.txt that match the training environment
- **Rule**: After training, immediately record exact package versions with `pip freeze`

### L7: numpy version must match between training and serving
- **Mistake**: Models saved with numpy 2.4.2 but requirements.txt had 1.26.4 → MT19937 BitGenerator error
- **Fix**: Update requirements.txt to numpy==2.4.2
- **Rule**: Same as L6 — pin all versions used during training

### L8: Mock joblib.load in CI to avoid model version dependency
- **Mistake**: CI tests failed because model files were saved with different library versions than installed
- **Fix**: Create `tests/conftest.py` that replaces `joblib.load` with mocks before any imports
- **Rule**: CI should never depend on real model files — always mock them

## Docker / Containerization

### L9: HF Spaces require port 7860
- **Mistake**: Dockerfile exposed port 8000 — HF Space crashed
- **Fix**: Use `EXPOSE 7860` and `${PORT:-7860}` in CMD
- **Rule**: HF Spaces only accept traffic on port 7860

### L10: Git LFS required for joblib model files
- **Mistake**: Tried to push .joblib files directly — HF rejected them as binary files too large
- **Fix**: `git lfs install` then `git lfs migrate import --include="*.joblib" --everything`
- **Rule**: Always track binary model files with Git LFS before first push

## GitHub Codespaces

### L11: Use `python3 -m pip` not `pip` in devcontainer
- **Mistake**: `postCreateCommand` used `pip` which isn't in PATH in the base image
- **Fix**: Use `python3 -m pip install` and `python3 -m uvicorn` everywhere
- **Rule**: Never assume `pip`, `uvicorn`, or `streamlit` are in PATH — always prefix with `python3 -m`

### L12: Rebuild Codespace after devcontainer.json changes
- **Mistake**: User's existing Codespace didn't pick up new devcontainer.json
- **Fix**: Ctrl+Shift+P → "Codespaces: Rebuild Container"
- **Rule**: Remind users to rebuild Codespace whenever devcontainer.json is changed

## Process

### L13: Diagnose before fixing
- **Mistake**: Jumped to fix HF API errors without reading logs first — user had to ask "find out what happened first"
- **Fix**: Always read logs/error output before proposing a solution
- **Rule**: No code changes until root cause is confirmed

### L14: Use plan mode for multi-step deployments
- **Mistake**: Made incremental fixes to deployment without a clear plan — caused multiple round trips
- **Fix**: Plan the full deployment sequence upfront (Dockerfile → README → push → verify)
- **Rule**: Any deployment task with 3+ steps requires plan mode first
