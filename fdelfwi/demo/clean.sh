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
    dotproduct_test
    dotproduct_lame_test
    gradient_test
    hessian_dp_test
    inversion_test
    mpi_inversion_test
    mpi_multishot_test
    multishot_test
    residual_test
    taylor_test
    trn_debug_test
    trn_inversion_test
    wave_op_dp_all
    wave_op_dp_freesurface
    wave_op_dp_test
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
