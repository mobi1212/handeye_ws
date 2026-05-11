# Repo Setup And Submodule Guide

## Overview

This repository uses an external SDK as a Git submodule:

- `src/anygrasp_sdk`

The main repository records which `anygrasp_sdk` commit should be used.
The SDK keeps its own Git history and remote separately.

## First-Time Clone

Clone the repository together with the submodule:

```bash
git clone --recurse-submodules git@github.com:mobi1212/handeye_ws.git
cd handeye_ws
```

If you already cloned the repository without submodules, run:

```bash
git submodule update --init --recursive
```

## Daily Update

To get the latest main repository changes:

```bash
git pull
git submodule update --init --recursive
```

If the main repository updates the recorded submodule commit, the second command
will move `src/anygrasp_sdk` to the expected version.

## Check Current State

Check the main repository:

```bash
git status
```

Check the submodule:

```bash
git -C src/anygrasp_sdk status
```

Check which submodule commit the main repository records:

```bash
git submodule status
```

## If You Change Only Main Repository Code

Example files:

- `src/ur3_handover/...`
- `docs/...`
- launch or config files in this repository

Workflow:

```bash
git add <files>
git commit -m "your message"
git push
```

## If You Change The Submodule

If you edit files inside `src/anygrasp_sdk`, you are editing a separate Git
repository.

Workflow:

```bash
git -C src/anygrasp_sdk status
git -C src/anygrasp_sdk add <files>
git -C src/anygrasp_sdk commit -m "your message"
git -C src/anygrasp_sdk push
```

Then update the main repository so it records the new submodule commit:

```bash
git add src/anygrasp_sdk
git commit -m "chore: update anygrasp_sdk submodule pointer"
git push
```

## Common Problems

### `src/anygrasp_sdk` looks modified in main repo

This usually means one of these:

- the submodule has uncommitted local changes
- the submodule HEAD moved to a different commit

Check:

```bash
git -C src/anygrasp_sdk status
git submodule status
```

### Submodule directory is empty after clone

Initialize it:

```bash
git submodule update --init --recursive
```

### Pull succeeds but code still looks old

Update the submodule after pull:

```bash
git submodule update --init --recursive
```

### Submodule commit cannot be fetched

This usually means the main repository points to a submodule commit that has not
been pushed to the submodule remote yet. The fix is to push the submodule first,
then pull again.

## Recommended Rule

When both repositories change, push in this order:

1. Push `src/anygrasp_sdk`
2. Push the main repository

That avoids broken submodule references for other users.
