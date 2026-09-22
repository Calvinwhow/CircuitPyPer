#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/Volumes/NIMLAB/DBS_Horn_Repository/HornDatasets/derivatives/leaddbs}"
DEST="${DEST:-/Volumes/HowExp/datasets/03z_howard_dbs}"
MODE="${MODE:-rsync}"     # Options are tar-copy | tar-file | rsync | tar-zst
ARCHIVE="${ARCHIVE:-$DEST/leaddbs_selected.tar}"
COMPRESS_PROGRAM="${COMPRESS_PROGRAM:-zstd -T0 --fast=3}"

mkdir -p "$DEST"

find_archive_paths() {
  local patient_dir="$1"
  local path

  for path in \
    "$patient_dir/reconstruction" \
    "$patient_dir/preprocessing"
  do
    [[ -d "$path" ]] && printf '%s\0' "$path"
  done
}

find_relative_archive_paths() {
  local patient_dir="$1"
  local patient
  local path

  patient="$(basename "$patient_dir")"
  for path in reconstruction preprocessing; do
    [[ -d "$patient_dir/$path" ]] && printf '%s\0' "$patient/$path"
  done
}

stream_tar_copy() {
  local list

  list="$(mktemp "${TMPDIR:-/tmp}/fast_archive.XXXXXX")"
  while IFS= read -r -d '' patient_dir; do
    find_relative_archive_paths "$patient_dir" >> "$list"
  done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0)

  if [[ ! -s "$list" ]]; then
    echo "No requested folders found under: $SRC"
    rm -f "$list"
    return 0
  fi

  COPYFILE_DISABLE=1 tar -C "$SRC" --null -T "$list" -cf - |
    COPYFILE_DISABLE=1 tar -C "$DEST" -xf -
  rm -f "$list"
}

stream_tar_archive() {
  local list
  local tmp

  list="$(mktemp "${TMPDIR:-/tmp}/fast_archive.XXXXXX")"
  tmp="${ARCHIVE}.tmp"
  while IFS= read -r -d '' patient_dir; do
    find_relative_archive_paths "$patient_dir" >> "$list"
  done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0)

  if [[ ! -s "$list" ]]; then
    echo "No requested folders found under: $SRC"
    rm -f "$list"
    return 0
  fi

  rm -f "$tmp"
  COPYFILE_DISABLE=1 tar -C "$SRC" --null -T "$list" -cf "$tmp"
  mv "$tmp" "$ARCHIVE"
  rm -f "$list"
  echo "$ARCHIVE"
}

copy_with_cp() {
  local folder
  local patient
  local patient_dir
  local dest_folder
  local src_folder

  while IFS= read -r -d '' patient_dir; do
    patient="$(basename "$patient_dir")"
    for folder in reconstruction preprocessing; do
      src_folder="$patient_dir/$folder"
      [[ -d "$src_folder" ]] || continue
      mkdir -p "$DEST/$patient"
      cp -Rp "$src_folder" "$DEST/$patient/"
    done
  done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0)
}

copy_with_rsync() {
  local folder
  local patient
  local patient_dir
  local src_folder

  while IFS= read -r -d '' patient_dir; do
    patient="$(basename "$patient_dir")"
    for folder in reconstruction preprocessing; do
      src_folder="$patient_dir/$folder"
      dest_folder="$DEST/$patient/$folder"

      # Make sure the folder exists in source and that it hasnt already been written
      [[ -d "$src_folder" ]] || continue
      [[ -d "$dest_folder" ]] && echo "exists: $dest_folder" && continue
      mkdir -p "$dest_folder"
      rsync -a --info=progress2 "$src_folder/" "$dest_folder/"
    done
  done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0)
}

archive_per_patient() {
  local list
  local out
  local patient
  local patient_dir
  local tmp

  while IFS= read -r -d '' patient_dir; do
    patient="$(basename "$patient_dir")"
    out="${DEST}/${patient}.tar.zst"
    tmp="${out}.tmp"
    list="${tmp}.files"

    if [[ -s "$out" ]]; then
      echo "Skipping completed patient: $patient"
      continue
    fi

    echo "Archiving patient: $patient"
    rm -f "$tmp" "$list"

    find_archive_paths "$patient_dir" > "$list"

    if [[ ! -s "$list" ]]; then
      echo "Skipping patient with no requested folders: $patient"
      rm -f "$list"
      continue
    fi

    if tar \
      --null \
      -T "$list" \
      --use-compress-program="$COMPRESS_PROGRAM" \
      -cf "$tmp"
    then
      mv "$tmp" "$out"
      echo "$out"
    else
      echo "Failed patient, skipping: $patient" >&2
      rm -f "$tmp"
    fi

    rm -f "$list"
  done < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0)
}

case "$MODE" in
  tar-copy)
    stream_tar_copy
    ;;
  tar-file)
    stream_tar_archive
    ;;
  cp)
    copy_with_cp
    ;;
  rsync)
    copy_with_rsync
    ;;
  tar-zst)
    archive_per_patient
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Use one of: tar-copy, tar-file, cp, rsync, tar-zst" >&2
    exit 2
    ;;
esac
