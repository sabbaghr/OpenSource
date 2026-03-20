#!/bin/bash
#
# Clean generated test output and macOS junk files
# Usage: ./clean.sh [all|dproduct|gradient|inversion]
#
DEMODIR=$(cd "$(dirname "$0")" && pwd)
REPODIR=$(cd "${DEMODIR}/../.." && pwd)

TARGET=${1:-all}

echo "=== Cleaning fdelfwi/demo test output (target: ${TARGET}) ==="

# --- dproduct/ work directories ---
DPRODUCT_DIRS=(
    dproduct/dotproduct_test
    dproduct/taylor_test
    dproduct/hessian_dp_test
    dproduct/wave_op_dp_test
    dproduct/wave_op_dp_all
    dproduct/wave_op_dp_freesurface
)

# --- gradient/ work directories ---
GRADIENT_DIRS=(
    gradient/gradient_visual_lame
    gradient/gradient_visual_velocity
    gradient/gradient_compare_scaling
)

# --- inversion/ work directories ---
INVERSION_DIRS=(
    inversion/lbfgs_inversion_lame
    inversion/lbfgs_inversion_velocity
    inversion/plbfgs_inversion_velocity
    inversion/trn_inversion_lame
    inversion/trn_inversion_velocity
)

clean_dirs() {
    local dirs=("$@")
    for d in "${dirs[@]}"; do
        if [ -d "${DEMODIR}/${d}" ]; then
            echo "  rm -rf ${d}/"
            rm -rf "${DEMODIR}/${d}"
        fi
    done
}

case "${TARGET}" in
    all)
        clean_dirs "${DPRODUCT_DIRS[@]}"
        clean_dirs "${GRADIENT_DIRS[@]}"
        clean_dirs "${INVERSION_DIRS[@]}"
        ;;
    dproduct)
        clean_dirs "${DPRODUCT_DIRS[@]}"
        ;;
    gradient)
        clean_dirs "${GRADIENT_DIRS[@]}"
        ;;
    inversion)
        clean_dirs "${INVERSION_DIRS[@]}"
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Usage: ./clean.sh [all|dproduct|gradient|inversion]"
        exit 1
        ;;
esac

echo "=== Cleaning macOS junk files from repo ==="
find "${REPODIR}" \( -name ".DS_Store" -o -name "._.DS_Store" -o -name "._*" \) -print -delete 2>/dev/null

echo "Done."
