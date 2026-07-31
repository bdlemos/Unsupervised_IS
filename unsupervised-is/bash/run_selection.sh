WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
cd "$WORKDIR"

datain="${DATASETS_DIR:-$WORKDIR/../datasets}"
out="${RESULTS_DIR:-$WORKDIR/../results}/instance_selection"

mkdir -p $out
mkdir -p "resources/logs"

datasets=(aisopos_ntua_2L)
methods=(bio-is)


for d in ${datasets[@]};
do
    echo $d ; 
    for method in ${methods[@]} 
    do
        echo $method ;
        python3 run\_generateSplit.py -d $d -m $method --datain $datain --out $out;
    done;
done;