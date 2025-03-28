#!/bin/bash

if [ "0$1" != "0dd" ] && [ "0$1" != "0rr" ] && [ "0$1" != "0tf" ]; then
  echo Please indicate whether to start with raw-reader [rr] or with DataDistribution [dd] or TfReader [tf] 1>&2
  exit 1
fi

if [[ -f local_env.sh ]]; then source ./local_env.sh; fi

export ALICE_O2_FST=1

if [[ -z "${WORKFLOW_PARAMETERS+x}" ]]; then
  export WORKFLOW_PARAMETERS="CALIB,QC,EVENT_DISPLAY,CALIB_LOCAL_AGGREGATOR"
  if [[ "0$FST_TMUX_INTEGRATED_AGGREGATOR" == "01" ]]; then
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},CALIB_LOCAL_INTEGRATED_AGGREGATOR"
  else
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},CALIB_PROXIES"
  fi
  [[ -z $ARGS_EXTRA_PROCESS_o2_eve_export_workflow ]] && export ARGS_EXTRA_PROCESS_o2_eve_export_workflow="--disable-write"
  if [[ -z "${GEN_TOPO_WORKDIR}" ]]; then
    mkdir -p gen_topo_tmp
    export GEN_TOPO_WORKDIR=`pwd`/gen_topo_tmp
  fi
fi
[[ -z "${SEVERITY}" ]] && export SEVERITY="important"

if [[ "0$FST_TMUX_NO_EPN" != "01" ]]; then
  # This sets up the hardcoded configuration to run the full system workflow on the EPN
  if [[ "0$EPN_NODE_MI100" == "01" && -z $EPN_GLOBAL_SCALING ]]; then
    export EPN_GLOBAL_SCALING="3 / 2"
  fi
  [[ -z $NGPUS ]] && export NGPUS=4
  [[ -z $GPUTYPE ]] && export GPUTYPE=HIP
  [[ -z $SHMSIZE ]] && export SHMSIZE=$(( (112 << 30) * ${EPN_GLOBAL_SCALING:-1} )) # Please keep these defaults in sync with those in shm-tool.sh
  [[ -z $DDSHMSIZE ]] && export DDSHMSIZE=$(( (112 << 10) * ${EPN_GLOBAL_SCALING:-1} ))
  [[ -z $GPUMEMSIZE ]] && export GPUMEMSIZE=$(( 24 << 30 ))
  [[ -z $NUMAGPUIDS ]] && export NUMAGPUIDS=1
  [[ -z $EPNPIPELINES ]] && export EPNPIPELINES=1
  [[ -z $O2_GPU_DOUBLE_PIPELINE ]] && export O2_GPU_DOUBLE_PIPELINE=1
  [[ -z $O2_GPU_RTC ]] && export O2_GPU_RTC=1
  [[ -z $DPL_CONDITION_BACKEND ]] && export DPL_CONDITION_BACKEND="http://localhost:8084"
  export ALL_EXTRA_CONFIG="$ALL_EXTRA_CONFIG;NameConf.mCCDBServer=${DPL_CONDITION_BACKEND};"
  export GEN_TOPO_QC_OVERRIDE_CCDB_SERVER="${DPL_CONDITION_BACKEND}"
  [[ -z $NUM_DPL_WORKFLOWS ]] && NUM_DPL_WORKFLOWS=2
  if [[ `lspci | grep "Vega 20\|Arcturus GL-XL" | wc -l` != "8" ]]; then
    echo "Could not detect 8 EPN GPUs, aborting" 1>&2
    exit 1
  fi
else
  [[ -z $NUM_DPL_WORKFLOWS ]] && NUM_DPL_WORKFLOWS=1
fi
export EXTINPUT=1
export SYNCMODE=1
export SHMTHROW=0
export IS_SIMULATED_DATA=1
export DATADIST_NEW_DPL_CHAN=1

[[ -z $GEN_TOPO_MYDIR ]] && GEN_TOPO_MYDIR="$(dirname $(realpath $0))"
source $GEN_TOPO_MYDIR/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }

workflow_has_parameter QC && export QC_REDIRECT_MERGER_TO_LOCALHOST=1

if [[ -z ${DPL_RAWPROXY_OVERRIDE_ORBITRESET+x} && $1 == "dd" ]]; then
  if [[ $BEAMTYPE == "PbPb" ]]; then
    export DPL_RAWPROXY_OVERRIDE_ORBITRESET=1550600800000
  else
    export DPL_RAWPROXY_OVERRIDE_ORBITRESET=1547590800000
  fi
fi

if [ "0$FST_TMUX_MEM_OVERRIDE" != "0" ]; then
  export SHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 30 ))
  export DDSHMSIZE=$(( $FST_TMUX_MEM_OVERRIDE << 10 ))
fi

if [ $1 == "dd" ]; then
  export CMD=datadistribution.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$(($NUM_DPL_WORKFLOWS + 3))
elif [ $1 == "tf" ]; then
  export CMD=tf-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$((NUM_DPL_WORKFLOWS + ${NUMAGPUIDS:-0}))
elif [ $1 == "rr" ]; then
  export CMD=raw-reader.sh
  export GPU_NUM_MEM_REG_CALLBACKS=$(($NUM_DPL_WORKFLOWS + ${NUMAGPUIDS:-0}))
fi

if [ "0$FST_TMUX_NOWAIT" != "01" ]; then
  ENDCMD="echo END; sleep 1000"
fi

if [ "0$FST_TMUX_KILLCHAINS" == "01" ]; then
  KILLCMD="sleep 60; ps aux | grep 'o2-dpl-run --session' | grep -v grep | awk '{print \$2}' | xargs kill -s INT --;"
fi

if [ "0$FST_TMUX_LOGPREFIX" != "0" ]; then
  LOGCMD=" &> ${FST_TMUX_LOGPREFIX}_[REPLACE].log"
fi

FST_SLEEP0=0
FST_SLEEP1=0
FST_SLEEP2=30
if [[ -z $SHM_MANAGER_SHMID ]]; then
  rm -f /dev/shm/*fmq*
  if [[ `ls /dev/shm/*fmq* 2> /dev/null | wc -l` != "0" ]]; then
    echo "FMQ SHM files left which cannot be deleted, please clean up!"
    exit 1
  fi
fi
[[ ${O2_GPU_RTC:-0} == 1 ]] && FST_SLEEP2=60
[[ ! -z $FST_TMUX_DD_WAIT ]] && FST_SLEEP2=$FST_TMUX_DD_WAIT

if workflow_has_parameter CALIB_PROXIES; then
  CALIB_COMMAND="$GEN_TOPO_MYDIR/aggregator-workflow.sh"
  if [[ -z ${CALIB_TASKS:-} ]]; then
    CALIB_TASKS="BARREL_TF CALO_TF BARREL_SPORADIC FORWARD_SPORADIC FORWARD_TF" # CALO_SPORADIC # Currently empty
    # [[ ${WORKFLOW_EXTRA_PROCESSING_STEPS:-} =~ (^|,)"MUON_SYNC_RECO"(,|$) ]] && CALIB_TASKS+=" MUON_TF MUON_SPORADIC" # Currently empty
  fi
else
  : ${CALIB_TASKS:=""}
fi

if [ "0$FST_TMUX_BATCH_MODE" == "01" ]; then
  { sleep $FST_SLEEP0; eval "NUMAID=0 $GEN_TOPO_MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/0}"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP1; eval "NUMAID=1 $GEN_TOPO_MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/1}"; eval "$ENDCMD"; } &
  { sleep $FST_SLEEP2; eval "SEVERITY=debug numactl --interleave=all $GEN_TOPO_MYDIR/$CMD ${LOGCMD/\[REPLACE]/2}"; eval "$KILLCMD $ENDCMD"; } &
  for i in $CALIB_TASKS; do
    { eval "AGGREGATOR_TASKS=$i $CALIB_COMMAND ${LOGCMD/\[REPLACE]/3_${i}}"; eval "$ENDCMD"; } &
  done
  wait
else
  TMUX_SPLIT_COMMAND="split-window"
  TMUX_COMMAND="tmux -L FST"
  TMUX_COMMAND+=" new-session  \"sleep $FST_SLEEP0; NUMAID=0 $GEN_TOPO_MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/0}; $ENDCMD\" ';'"
  for i in `seq 1 $(($NUM_DPL_WORKFLOWS - 1))`; do
    TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"sleep $FST_SLEEP1; NUMAID=$i $GEN_TOPO_MYDIR/dpl-workflow.sh ${LOGCMD/\[REPLACE]/1}; $ENDCMD\" ';'"
  done
  TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"sleep $FST_SLEEP2; SEVERITY=debug numactl --interleave=all $GEN_TOPO_MYDIR/$CMD; $KILLCMD $ENDCMD\" ';'"
  FIRST_CALIB=1
  for i in $CALIB_TASKS; do
    TMUX_COMMAND+=" $TMUX_SPLIT_COMMAND \"AGGREGATOR_TASKS=$i $CALIB_COMMAND ${LOGCMD/\[REPLACE]/3_${i}}; $ENDCMD\" ';'"
    if [[ $FIRST_CALIB == 1 ]]; then
      TMUX_COMMAND+=" select-layout even-vertical ';'"
      TMUX_SPLIT_COMMAND="split-window -h"
      FIRST_CALIB=0
    fi
  done
  [[ $FIRST_CALIB == 1 ]] && TMUX_COMMAND+=" select-layout even-vertical ';'"
  # echo "Running $TMUX_COMMAND"
  eval $TMUX_COMMAND
fi

if [[ -z $SHM_MANAGER_SHMID ]]; then
  echo "Cleaning up SHM"
  rm -f /dev/shm/*fmq*
fi
