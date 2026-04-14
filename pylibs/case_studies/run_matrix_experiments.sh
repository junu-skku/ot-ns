#!/bin/bash

# 에러 발생 시 즉시 종료
set -e

# 스크립트가 위치한 디렉토리 및 실행할 파이썬 스크립트 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/mle_trickle_vicious_cycle_0414.py"

# 결과를 저장할 최상위 디렉토리
BASE_OUT_DIR="tmp/matrix_results"

# 실험할 변수 배열 (총 12가지 조합)
NUM_TOTAL=(1 2 3 4 5 6 7 8 9 10 11 12)
PAYLOAD_SIZES=(50 100 150 200)
GRID_SPACINGS=(20 30 40)

# 최상위 디렉토리 생성 및 통합(Master) 로그 파일 준비
mkdir -p "$BASE_OUT_DIR"
MASTER_LOG="$BASE_OUT_DIR/master_summary.log"

echo "========================================================" > "$MASTER_LOG"
echo "Matrix Experiments Master Log" >> "$MASTER_LOG"
echo "Started at: $(date)" >> "$MASTER_LOG"
echo "========================================================" >> "$MASTER_LOG"

echo "========================================================"
echo "Starting Matrix Experiments"
echo "Payload Sizes: ${PAYLOAD_SIZES[*]}"
echo "Grid Spacings: ${GRID_SPACINGS[*]}"
echo "Total combinations: $(( ${#PAYLOAD_SIZES[@]} * ${#GRID_SPACINGS[@]} ))"
echo "Master log will be saved to: $MASTER_LOG"
echo "========================================================"

for i in "${NUM_TOTAL[@]}"; do
    for spacing in "${GRID_SPACINGS[@]}"; do
        for size in "${PAYLOAD_SIZES[@]}"; do
            OUT_DIR="$BASE_OUT_DIR/run_${i}_spacing_${spacing}_payload_${size}"
            RUN_LOG="$OUT_DIR/run.log"
            
            echo "[*] Running experiment with Grid Spacing = $spacing, Payload Size = $size"
            echo "    Output Directory: $OUT_DIR"
            
            # 해당 테스트 케이스를 위한 폴더 생성
            mkdir -p "$OUT_DIR"
            
            # 파이썬 스크립트 실행 결과를 터미널에 출력하면서 동시에 개별 run.log에 저장
            # 파이프라인 중 에러 발생 시 스크립트가 멈추도록 설정
            set -o pipefail
            python3 "$PYTHON_SCRIPT" \
                --grid-spacing "$spacing" \
                --coap-payload-size "$size" \
                --output-dir "$OUT_DIR" 2>&1 | tee "$RUN_LOG"
            set +o pipefail
            
            # 통합 로그에 현재 테스트 케이스 결과를 명확한 구분선과 함께 추가
            {
                echo ""
                echo ""
                echo "################################################################################"
                echo "### [RESULT] Grid Spacing: $spacing | Payload Size: $size"
                echo "### Directory: $OUT_DIR"
                echo "################################################################################"
                cat "$RUN_LOG"
            } >> "$MASTER_LOG"
                
            echo "[+] Finished experiment: Spacing=$spacing, Payload=$size"
            echo "--------------------------------------------------------"
        done
    done
done

echo "All experiments completed successfully."
echo "Results are saved in: $BASE_OUT_DIR"
echo "Unified Master Log is available at: $MASTER_LOG"
