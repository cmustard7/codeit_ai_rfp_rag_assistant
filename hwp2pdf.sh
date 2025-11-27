set -e
# $1: 입력 HWP/HWPX 파일, $2: 출력 경로 (예: /tmp/out.pdf)
soffice --headless --convert-to pdf --outdir "$(dirname "$2")" "$1"
