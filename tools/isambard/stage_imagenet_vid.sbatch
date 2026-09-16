#!/bin/bash
# Verify, extract and check the ImageNet VID + DET data on Isambard-AI.
#
# Input, uploaded beforehand to $ROOT/archives:
#     ILSVRC2015_VID.tar.gz  ILSVRC2017_DET.tar.gz.aa  ILSVRC2017_DET.tar.gz.ab  annotations.tar.gz
# Output, $ROOT/ILSVRC in the layout the configs expect as data_root "data/ILSVRC/":
#     annotations/*.json  Annotations/{VID,DET}  Data/{VID,DET}  ImageSets/{VID,DET}
#
# Submit from the repo root on a login node:
#     sbatch tools/isambard/stage_imagenet_vid.sbatch
#
# Runs as a job, not on the login node: extraction creates ~1.6M files on Lustre.
# Isambard-AI only allocates through GPUs, so this asks for one and uses the 72
# CPU cores that come with it.
#
#SBATCH --job-name=stage_imagenet_vid
#SBATCH --account=brics.b5cs
#SBATCH --output=stage_imagenet_vid_%j.out
#SBATCH --gpus=1
#SBATCH --time=03:00:00
set -euo pipefail

ROOT="${ROOT:-/projects/b5cs/imagenet_vid}"
ARCH="$ROOT/archives"
DEST="$ROOT/ILSVRC"
REPO="${REPO:-$HOME/code/vfe.pytorch}"
PY="${PY:-$REPO/.venv/bin/python}"

# ---- 1. checksums ----------------------------------------------------------
# SHA-256 of each file as published in the Hugging Face dataset repo
# guanxiongsun/imagenetvid (its LFS object ids). The VID archive was uploaded as
# one file; HF splits it at 48318382080 bytes, so it is checked as those parts.
# A match proves the upload is intact *and* identical to the HF copy.
VID_SPLIT=48318382080
declare -A EXPECTED=(
    [vid.aa]=36b8f27c8cfd3d3a55ea34499c48dbb99946ebfc0a20fa6918c2fc3f76788fae
    [vid.ab]=c1e3082d6f18fdd0dc42210618bd67cc7de8dfe3c7337278e4c53c0660c8c971
    [det.aa]=a913063c02ff1d6d5916f44c5377d73229d28f852f064adbfe3580736229b0c7
    [det.ab]=8b402dcc94aabe165625328dda0dd8e16c93eadb9586680b737e20c7c091a3b7
    [annotations]=50fe6e7ef3bdcf9dcf3653d84110c189df648a46dfbe874f0792f0c4bd3a4689
)

echo "==> $(date -Is) verifying checksums"
SUMS="$(mktemp -d "$ROOT/.sha256.XXXXXX")"   # not /tmp: node-local on Isambard
head -c "$VID_SPLIT" "$ARCH/ILSVRC2015_VID.tar.gz" | sha256sum | cut -d' ' -f1 > "$SUMS/vid.aa" &
tail -c +"$((VID_SPLIT + 1))" "$ARCH/ILSVRC2015_VID.tar.gz" | sha256sum | cut -d' ' -f1 > "$SUMS/vid.ab" &
sha256sum < "$ARCH/ILSVRC2017_DET.tar.gz.aa" | cut -d' ' -f1 > "$SUMS/det.aa" &
sha256sum < "$ARCH/ILSVRC2017_DET.tar.gz.ab" | cut -d' ' -f1 > "$SUMS/det.ab" &
sha256sum < "$ARCH/annotations.tar.gz" | cut -d' ' -f1 > "$SUMS/annotations" &
# A pipeline that failed leaves an empty or wrong checksum, caught below.
wait

bad=0
for part in vid.aa vid.ab det.aa det.ab annotations; do
    got="$(cat "$SUMS/$part")"
    if [[ "$got" == "${EXPECTED[$part]}" ]]; then
        echo "    sha256 ok        $part"
    else
        echo "    sha256 MISMATCH  $part: got '$got'"
        bad=1
    fi
done
rm -rf "$SUMS"
if (( bad )); then
    echo "checksum mismatch; not extracting" >&2
    exit 1
fi

# ---- 2. extract ------------------------------------------------------------
# Skipped: the VID video snippets and the VID/DET test splits. They have no
# labels and no config reads them; the archives above still hold them.
# Patterns are wildcards so they match names before and after stripping the
# archives' top-level directory (ILSVRC2015/ for VID, ILSVRC/ for DET).
EXCLUDES=(--wildcards --exclude='*Data/VID/snippets*' --exclude='*Data/VID/test*'
          --exclude='*Data/DET/test*')

echo "==> $(date -Is) extracting into $DEST"
mkdir -p "$DEST"
( pigz -dc "$ARCH/ILSVRC2015_VID.tar.gz" \
    | tar -x -C "$DEST" --strip-components=1 "${EXCLUDES[@]}" ) &
vid_pid=$!
( cat "$ARCH/ILSVRC2017_DET.tar.gz.aa" "$ARCH/ILSVRC2017_DET.tar.gz.ab" | pigz -dc \
    | tar -x -C "$DEST" --strip-components=1 "${EXCLUDES[@]}" ) &
det_pid=$!
tar -xzf "$ARCH/annotations.tar.gz" -C "$DEST"
wait "$vid_pid"
wait "$det_pid"

# ---- 3. completeness -------------------------------------------------------
echo "==> $(date -Is) checking every annotated image exists"
"$PY" "$REPO/tools/isambard/check_imagenet_vid.py" --root "$DEST" --workers 64 \
    | tee "$ROOT/CHECK.txt"

cat > "$ROOT/MANIFEST.txt" <<EOF
ImageNet VID + DET for vfe.pytorch, staged $(date -Is) by job ${SLURM_JOB_ID:-?}.
archives/  uploaded from the local copy; SHA-256 verified against the
           Hugging Face dataset repo guanxiongsun/imagenetvid.
ILSVRC/    extracted from archives/, without Data/VID/snippets, Data/VID/test
           and Data/DET/test. Use as data_root (configs: "data/ILSVRC/").
CHECK.txt  every image referenced by annotations/*.json exists.
Not backed up: Isambard storage is working storage, removed at project end.
EOF
echo "==> $(date -Is) done"
