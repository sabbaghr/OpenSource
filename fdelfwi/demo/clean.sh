#!/bin/bash
#
# Clean generated test output and macOS junk files
# Usage: ./clean.sh
#
DEMODIR=$(cd "$(dirname "$0")" && pwd)
REPODIR=$(cd "${DEMODIR}/../.." && pwd)

echo "=== Cleaning fdelfwi/demo test output ==="

# Remove test output directories (created by test_*.sh scripts)
WORKDIRS=(
    # Tier 1 - Core physics tests
    wave_op_dp_test
    wave_op_dp_all
    wave_op_dp_freesurface
    dotproduct_test
    dotproduct_nodecim
    dotproduct_small
    taylor_test
    hessian_dp_test
    # Tier 2 - Optimization tests
    optimizer_test
    scaling_test
    # Tier 3 - Inversion tests
    lbfgs_inversion_test
    trn_inversion_test
    gradient_visual_test
)

for d in "${WORKDIRS[@]}"; do
    if [ -d "${DEMODIR}/${d}" ]; then
        echo "  rm -rf ${d}/"
        rm -rf "${DEMODIR}/${d}"
    fi
done

echo "=== Cleaning macOS junk files from repo ==="
find "${REPODIR}" \( -name ".DS_Store" -o -name "._.DS_Store" -o -name "._*" \) -print -delete 2>/dev/null

echo "Done."
