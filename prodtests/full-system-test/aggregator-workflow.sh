#!/bin/bash

# Non-zero exit code already if one command in a pipe fails
set -o pipefail

# Abort in case any variable is not bound
if [[ ${IGNORE_UNBOUND_VARIABLES:-} != 1 ]]; then  set -u; fi

#SEVERITY="detail"
#ENABLE_METRICS=1
[ -d "${O2DPG_ROOT:-}" ] || { echo "O2DPG_ROOT not set" 1>&2; exit 1; }

source $O2DPG_ROOT/DATA/common/setenv.sh || { echo "setenv.sh failed" 1>&2 && exit 1; }
source $O2DPG_ROOT/DATA/common/getCommonArgs.sh || { echo "getCommonArgs.sh failed" 1>&2 && exit 1; }
source $O2DPG_ROOT/DATA/common/setenv_calib.sh || { echo "setenv_calib.sh failed" 1>&2 && exit 1; }

# if the populator for DCS CCDB is needed, set it to non-0
: ${NEED_DCS_CCDB_POPULATOR:=0}

# the production CCDB populator will accept subspecs in this range
: ${CCDBPRO_SUBSPEC_MIN:=0}
: ${CCDBPRO_SUBSPEC_MAX:=32767}

# the DCS CCDB populator will accept subspecs in this range
: ${CCDBDCS_SUBSPEC_MIN:=32768}
: ${CCDBDCS_SUBSPEC_MAX:=65535}

# check that WORKFLOW_DETECTORS is needed, otherwise the wrong calib wf will be built
if [[ -z ${WORKFLOW_DETECTORS:-} ]]; then echo "WORKFLOW_DETECTORS must be defined" 1>&2; exit 1; fi

# CCDB destination for uploads to the standard ccdb
if [[ -z ${CCDB_POPULATOR_UPLOAD_PATH+x} ]]; then
  if [[ $RUNTYPE == "SYNTHETIC" || "${GEN_TOPO_DEPLOYMENT_TYPE:-}" == "ALICE_STAGING" ]]; then
    CCDB_POPULATOR_UPLOAD_PATH="http://ccdb-test.cern.ch:8080"
  elif [[ $RUNTYPE == "PHYSICS" ]]; then
    if [[ $EPNSYNCMODE == 1 ]]; then
      CCDB_POPULATOR_UPLOAD_PATH="http://localhost:8084"
    else
      CCDB_POPULATOR_UPLOAD_PATH="http://ccdb-test.cern.ch:8080"
    fi
  else
    CCDB_POPULATOR_UPLOAD_PATH="none"
  fi
fi
# CCDB destination for uploads to the DCS exchange ccdb
if [[ -z ${CCDB_DCS_POPULATOR_UPLOAD_PATH+x} ]]; then
  if [[ $RUNTYPE == "SYNTHETIC" || "${GEN_TOPO_DEPLOYMENT_TYPE:-}" == "ALICE_STAGING" ]]; then
    CCDB_DCS_POPULATOR_UPLOAD_PATH="http://ccdb-test.cern.ch:8080"
  elif [[ $RUNTYPE == "PHYSICS" ]]; then
    if [[ $EPNSYNCMODE == 1 ]]; then
      CCDB_DCS_POPULATOR_UPLOAD_PATH="$DCSCCDBSERVER_PERS"
    else
      CCDB_DCS_POPULATOR_UPLOAD_PATH="http://ccdb-test.cern.ch:8080"
    fi
  else
    CCDB_DCS_POPULATOR_UPLOAD_PATH="none"
  fi
fi
if [[ "${GEN_TOPO_VERBOSE:-}" == "1" ]]; then
  echo "CCDB_POPULATOR_UPLOAD_PATH = $CCDB_POPULATOR_UPLOAD_PATH" 1>&2
  echo "CCDB_DCS_POPULATOR_UPLOAD_PATH = $CCDB_DCS_POPULATOR_UPLOAD_PATH" 1>&2
fi

# Avoid writing calibration data for run types different than physics
if [[ $RUNTYPE != "PHYSICS" ]] && [[ $CALIB_DIR == "/data/calibration" ]]; then
  if [[ ${FORCE_LOCAL_CALIBRATION_OUTPUT:-} != 1 ]]; then
    export CALIB_DIR="/dev/null"
  else
    # Special setting to allow for expert tests. In this case output is written to the current working directory
    # Since in this case also a meta file would be written we need to disable that explicitly
    export CALIB_DIR=$FILEWORKDIR
    export EPN2EOS_METAFILES_DIR="/dev/null"
  fi
fi


# Adding calibrations
EXTRA_WORKFLOW_CALIB=

if [[ "${GEN_TOPO_VERBOSE:-}" == "1" ]]; then
  echo "CALIB_PRIMVTX_MEANVTX = $CALIB_PRIMVTX_MEANVTX" 1>&2
  echo "CALIB_TOF_LHCPHASE = $CALIB_TOF_LHCPHASE" 1>&2
  echo "CALIB_TOF_CHANNELOFFSETS = $CALIB_TOF_CHANNELOFFSETS" 1>&2
  echo "CALIB_TOF_DIAGNOSTICS = $CALIB_TOF_DIAGNOSTICS" 1>&2
  echo "CALIB_EMC_BADCHANNELCALIB = $CALIB_EMC_BADCHANNELCALIB" 1>&2
  echo "CALIB_EMC_TIMECALIB = $CALIB_EMC_TIMECALIB" 1>&2
  echo "CALIB_PHS_ENERGYCALIB = $CALIB_PHS_ENERGYCALIB" 1>&2
  echo "CALIB_PHS_BADMAPCALIB = $CALIB_PHS_BADMAPCALIB" 1>&2
  echo "CALIB_PHS_TURNONCALIB = $CALIB_PHS_TURNONCALIB" 1>&2
  echo "CALIB_PHS_RUNBYRUNCALIB = $CALIB_PHS_RUNBYRUNCALIB" 1>&2
  echo "CALIB_PHS_L1PHASE = $CALIB_PHS_L1PHASE" 1>&2
  echo "CALIB_TRD_VDRIFTEXB = $CALIB_TRD_VDRIFTEXB" 1>&2
  echo "CALIB_TRD_GAIN = $CALIB_TRD_GAIN" 1>&2
  echo "CALIB_TPC_TIMEGAIN = $CALIB_TPC_TIMEGAIN" 1>&2
  echo "CALIB_TPC_RESPADGAIN = $CALIB_TPC_RESPADGAIN" 1>&2
  echo "CALIB_TPC_SCDCALIB = $CALIB_TPC_SCDCALIB" 1>&2
  echo "CALIB_TPC_VDRIFTTGL = $CALIB_TPC_VDRIFTTGL" 1>&2
  echo "CALIB_TPC_IDC = $CALIB_TPC_IDC" 1>&2
  echo "CALIB_TPC_SAC = $CALIB_TPC_SAC" 1>&2
  echo "CALIB_CPV_GAIN = $CALIB_CPV_GAIN" 1>&2
  echo "CALIB_ZDC_TDC = $CALIB_ZDC_TDC" 1>&2
  echo "CALIB_FT0_TIMEOFFSET = $CALIB_FT0_TIMEOFFSET" 1>&2
  echo "CALIB_ITS_DEADMAP_TIME = $CALIB_ITS_DEADMAP_TIME" 1>&2
  echo "CALIB_MFT_DEADMAP_TIME = $CALIB_MFT_DEADMAP_TIME" 1>&2
  echo "CALIB_RCT_UPDATER = ${CALIB_RCT_UPDATER:-}" 1>&2
fi

# beamtype dependent settings
: ${FT0_TIMEOFFSET_TF_PER_SLOT:=105600}
: ${INTEGRATEDCURR_TF_PER_SLOT:=150000} # setting for FT0, FV0, FDD and TOF

if [[ $BEAMTYPE == "PbPb" ]]; then
  : ${LHCPHASE_TF_PER_SLOT:=100000}
  : ${TOF_CHANNELOFFSETS_UPDATE:=300000}
  : ${TOF_CHANNELOFFSETS_DELTA_UPDATE:=50000}
  : ${FT0_TIMEOFFSET_TRG_BITS:=384} # min bias and data validity
else
  : ${LHCPHASE_TF_PER_SLOT:=100000}
  : ${TOF_CHANNELOFFSETS_UPDATE:=300000}
  : ${TOF_CHANNELOFFSETS_DELTA_UPDATE:=50000}
  : ${FT0_TIMEOFFSET_TRG_BITS:=144} # vertex and data validity
fi

# special settings for aggregator workflows
if [[ "${CALIB_TPC_SCDCALIB_SENDTRKDATA:-}" == "1" ]]; then ENABLE_TRACK_INPUT="--enable-track-input"; else ENABLE_TRACK_INPUT=""; fi

# Calibration workflows
if ! workflow_has_parameter CALIB_LOCAL_INTEGRATED_AGGREGATOR; then
  WORKFLOW=
  [[ "${GEN_TOPO_ONTHEFLY:-}" == "1" ]] && WORKFLOW="echo '{}' | " # When running in a pseudo terminal / with ODC, sometimes we have bogus stdin file descriptors
else
  : ${AGGREGATOR_TASKS:=ALL}
fi

if [[ -z ${AGGREGATOR_TASKS:-} ]]; then
  echo "ERROR: AGGREGATOR_TASKS is empty, you need to define it to run!" 1>&2
  exit 1
fi

if [[ "${GEN_TOPO_VERBOSE:-}" == "1" ]]; then
  # Which calibrations are we aggregating
  echo "AGGREGATOR_TASKS = $AGGREGATOR_TASKS" 1>&2
fi

# adding input proxies
if workflow_has_parameter CALIB_PROXIES; then
  if [[ $AGGREGATOR_TASKS == BARREL_TF ]]; then
    if [[ ! -z ${CALIBDATASPEC_BARREL_TF:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_BARREL_TF\" $(get_proxy_connection barrel_tf input timeframe)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == BARREL_SPORADIC ]]; then
    if [[ ! -z ${CALIBDATASPEC_BARREL_SPORADIC:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_BARREL_SPORADIC\" $(get_proxy_connection barrel_sp input sporadic)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == TPC_IDCBOTH_SAC ]]; then
    if [[ $EPNSYNCMODE != 1 ]]; then
      echo "ERROR: TPC IDC / SAC calib workflow enabled without EPNSYNCMODE, please note that there will not be input data for it" 1>&2
    fi
    CHANNELS_LIST=
    [[ $EPNSYNCMODE == 0 ]] && FLP_ADDRESS="tcp://localhost:29950"
    if [[ ! -z ${CALIBDATASPEC_TPCIDC_A:-} ]] || [[ ! -z ${CALIBDATASPEC_TPCIDC_C:-} ]]; then
      # define port for FLP
      : ${TPC_IDC_FLP_PORT:=29950}
      # expand FLPs; TPC uses from 001 to 145, but 145 is reserved for SAC
      if [[ "${GEN_TOPO_DEPLOYMENT_TYPE:-}" == "ALICE_STAGING" ]]; then
        FLP_ADDRESS="tcp://alio2-cr1-mvs03-ib:${TPC_IDC_FLP_PORT}"
        CHANNELS_LIST+="type=pull,name=tpcidc_flp,transport=zeromq,address=$FLP_ADDRESS,method=connect,rateLogging=10;"
      else
        for flp in $(seq -f "%03g" 1 144); do
          [[ ! $FLP_IDS =~ (^|,)"$flp"(,|$) ]] && continue
          [[ $EPNSYNCMODE == 1 ]] && FLP_ADDRESS="tcp://alio2-cr1-flp${flp}-ib:${TPC_IDC_FLP_PORT}"
          CHANNELS_LIST+="type=pull,name=tpcidc_flp${flp},transport=zeromq,address=$FLP_ADDRESS,method=connect,rateLogging=10;"
        done
      fi
    fi
    if [[ ! -z ${CALIBDATASPEC_TPCSAC:-} ]]; then
      # define port for FLP
      [[ -z ${TPC_SAC_FLP_PORT:-} ]] && TPC_SAC_FLP_PORT=29951
      [[ $EPNSYNCMODE == 1 ]] && FLP_ADDRESS="tcp://alio2-cr1-flp145-ib:${TPC_SAC_FLP_PORT}"
      CHANNELS_LIST+="type=pull,name=tpcidc_sac,transport=zeromq,address=$FLP_ADDRESS,method=connect,rateLogging=10;"
    fi
    if [[ ! -z $CHANNELS_LIST ]]; then
      DATASPEC_LIST=
      if [[ ! -z ${CALIBDATASPEC_TPCIDC_A:-} ]]; then
        add_semicolon_separated DATASPEC_LIST "\"$CALIBDATASPEC_TPCIDC_A\""
      fi
      if [[ ! -z ${CALIBDATASPEC_TPCIDC_C:-} ]]; then
        add_semicolon_separated DATASPEC_LIST "\"$CALIBDATASPEC_TPCIDC_C\""
      fi
      if [[ ! -z ${CALIBDATASPEC_TPCSAC:-} ]]; then
        add_semicolon_separated DATASPEC_LIST "\"$CALIBDATASPEC_TPCSAC\""
      fi
      add_W o2-dpl-raw-proxy "--proxy-name tpcidc --io-threads 2 --dataspec \"$DATASPEC_LIST\" --sporadic-outputs --channel-config \"$CHANNELS_LIST\" ${TIMEFRAME_SHM_LIMIT+--timeframes-shm-limit} $TIMEFRAME_SHM_LIMIT" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == CALO_TF ]]; then
    if [[ ! -z ${CALIBDATASPEC_CALO_TF:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_CALO_TF\" $(get_proxy_connection calo_tf input timeframe)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == CALO_SPORADIC ]]; then
    if [[ ! -z ${CALIBDATASPEC_CALO_SPORADIC:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_CALO_SPORADIC\" $(get_proxy_connection calo_sp input sporadic)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == MUON_TF ]]; then
    if [[ ! -z ${CALIBDATASPEC_MUON_TF:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_MUON_TF\" $(get_proxy_connection muon_tf input timeframe)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == MUON_SPORADIC ]]; then
    if [[ ! -z ${CALIBDATASPEC_MUON_SPORADIC:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_MUON_SPORADIC\" $(get_proxy_connection muon_sp input sporadic)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == FORWARD_TF ]]; then
    if [[ ! -z ${CALIBDATASPEC_FORWARD_TF:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_FORWARD_TF\" $(get_proxy_connection fwd_tf input timeframe)" "" 0
    fi
  elif [[ $AGGREGATOR_TASKS == FORWARD_SPORADIC ]]; then
    if [[ ! -z ${CALIBDATASPEC_FORWARD_SPORADIC:-} ]]; then
      add_W o2-dpl-raw-proxy "--dataspec \"$CALIBDATASPEC_FORWARD_SPORADIC\" $(get_proxy_connection fwd_sp input sporadic)" "" 0
    fi
  fi
fi

# calibrations for AGGREGATOR_TASKS == BARREL_TF
if [[ $AGGREGATOR_TASKS == BARREL_TF ]] || [[ $AGGREGATOR_TASKS == ALL ]]; then
  # RCT updater
  if [[ ${CALIB_RCT_UPDATER:-} == 1 ]]; then
    [[ -z ${TDIFFCHECK:-} ]] && [[ $EPNSYNCMODE == 1 ]] && TDIFFCHECK=15000
    add_W o2-rct-updater-workflow "--ccdb-server $CCDB_POPULATOR_UPLOAD_PATH --max-diff-orbit-creationtime ${TDIFFCHECK:--1}"
  fi
  # PrimaryVertex
  if [[ $CALIB_PRIMVTX_MEANVTX == 1 ]]; then
    : ${TFPERSLOTS_MEANVTX:=55000}
    : ${DELAYINTFS_MEANVTX:=10}
    : ${DCSSUBSPEC_MEANVTX:=$CCDBDCS_SUBSPEC_MIN}   # set 0 to deactivate sending CSV meanvertex version, use value >= $CCDBDCS_SUBSPEC_MIN to send it to DCS CCDB server instead of production one
    [[ $DCSSUBSPEC_MEANVTX -ge $CCDBDCS_SUBSPEC_MIN ]] && NEED_DCS_CCDB_POPULATOR=1
    add_W o2-calibration-mean-vertex-calibration-workflow "--meanvertex-dcs-subspec $DCSSUBSPEC_MEANVTX" "MeanVertexCalib.tfPerSlot=$TFPERSLOTS_MEANVTX;MeanVertexCalib.maxTFdelay=$DELAYINTFS_MEANVTX"
  fi
  # ITS
  if [[ $CALIB_ITS_DEADMAP_TIME == 1 ]]; then
     add_W o2-itsmft-deadmap-builder-workflow "--ccdb-url $CCDB_POPULATOR_UPLOAD_PATH ${CALIB_ITS_DEADMAP_TIME_OPT:-}"
  fi
  # MFT
  if [[ $CALIB_MFT_DEADMAP_TIME == 1 ]]; then
     add_W o2-itsmft-deadmap-builder-workflow  "--runmft --ccdb-url $CCDB_POPULATOR_UPLOAD_PATH ${CALIB_MFT_DEADMAP_TIME_OPT:---skip-static-map}"
  fi
  # TOF
  if [[ $CALIB_TOF_LHCPHASE == 1 ]] || [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
    if [[ $CALIB_TOF_LHCPHASE == 1 ]]; then
      add_W o2-calibration-tof-calib-workflow "--do-lhc-phase --tf-per-slot $LHCPHASE_TF_PER_SLOT --use-ccdb --max-delay 0 " "" 0
    fi
    if [[ $CALIB_TOF_CHANNELOFFSETS == 1 ]]; then
      add_W o2-calibration-tof-calib-workflow "--do-channel-offset --update-interval $TOF_CHANNELOFFSETS_UPDATE --delta-update-interval $TOF_CHANNELOFFSETS_DELTA_UPDATE --min-entries 100 --range 100000 --use-ccdb --condition-tf-per-query 2640 " "" 0
    fi
  fi
  if [[ $CALIB_TOF_DIAGNOSTICS == 1 ]]; then
    add_W o2-calibration-tof-diagnostic-workflow "--tf-per-slot $LHCPHASE_TF_PER_SLOT --max-delay 1" "" 0
  fi
  # TPC
  if [[ $CALIB_TPC_SCDCALIB == 1 ]]; then
    add_W o2-calibration-residual-aggregator "--disable-root-input ${CALIB_TPC_SCDCALIB_SLOTLENGTH:+"--sec-per-slot $CALIB_TPC_SCDCALIB_SLOTLENGTH"} $ENABLE_TRACK_INPUT $CALIB_TPC_SCDCALIB_CTP_INPUT --output-dir $CALIB_DIR --meta-output-dir $EPN2EOS_METAFILES_DIR ${RESIDUAL_AGGREGATOR_AUTOSAVE:+"--autosave-interval $RESIDUAL_AGGREGATOR_AUTOSAVE"}"
  fi
  if [[ $CALIB_TPC_VDRIFTTGL == 1 ]]; then
    # options available via ARGS_EXTRA_PROCESS_o2_tpc_vdrift_tgl_calibration_workflow="--nbins-tgl 20 --nbins-dtgl 50 --max-tgl-its 2. --max-dtgl-itstpc 0.15 --min-entries-per-slot 1000 --time-slot-seconds 600 <--vdtgl-histos-file-name name> "
    add_W o2-tpc-vdrift-tgl-calibration-workflow ""
  fi
  # TRD
  TRD_CALIB_CONFIG=
  if [[ $CALIB_TRD_VDRIFTEXB == 1 ]]; then
    TRD_CALIB_CONFIG+=" --vDriftAndExB"
  fi
  if [[ $CALIB_TRD_GAIN == 1 ]]; then
    TRD_CALIB_CONFIG+=" --gain"
  fi
  if [[ $CALIB_TRD_T0 == 1 ]]; then
    TRD_CALIB_CONFIG+=" --t0"
  fi
  if [[ ! -z ${TRD_CALIB_CONFIG} ]]; then
    add_W o2-calibration-trd-workflow "${TRD_CALIB_CONFIG}"
  fi
fi

# calibrations for AGGREGATOR_TASKS == BARREL_SPORADIC
if [[ $AGGREGATOR_TASKS == BARREL_SPORADIC ]] || [[ $AGGREGATOR_TASKS == ALL ]]; then
  # TPC
  if [[ $CALIB_TPC_TIMEGAIN == 1 ]]; then
    add_W o2-tpc-calibrator-dedx "--min-entries-sector 3000 --min-entries-1d 200 --min-entries-2d 10000"
  fi
  if [[ $CALIB_TPC_RESPADGAIN == 1 ]]; then
    add_W o2-tpc-calibrator-gainmap-tracks "--tf-per-slot 200000 --store-RMS-CCDB true"
  fi
  # TOF
  if [[ $CALIB_TOF_INTEGRATEDCURR == 1 ]]; then
    add_W o2-tof-merge-integrate-cluster-workflow "--tf-per-slot $INTEGRATEDCURR_TF_PER_SLOT"
  fi
fi

# TPC IDCs and SAC
crus="0-359"  # to be used with $AGGREGATOR_TASKS == TPC_IDCBOTH_SAC or ALL
lanesFactorize=${O2_TPC_IDC_FACTORIZE_NLANES:-12}
threadFactorize=${O2_TPC_IDC_FACTORIZE_NTHREADS:-16}
nTFs=$((1000 * 128 / ${NHBPERTF}))
nTFs_SAC=$((10000 * 128 / ${NHBPERTF}))
nBuffer=$((100 * 128 / ${NHBPERTF}))
IDC_DELTA="--disable-IDCDelta true" # off by default
# deltas are on by default; you need to request explicitly to switch them off;
if [[ "${DISABLE_IDC_DELTA:-}" == "1" ]]; then IDC_DELTA=""; fi
if [[ "${ENABLE_IDC_DELTA_FILE:-}" == "1" ]]; then IDC_DELTA+=" --dump-IDCDelta-calib-data true --output-dir $CALIB_DIR --meta-output-dir $EPN2EOS_METAFILES_DIR "; fi

if [[ "${DISABLE_IDC_PAD_MAP_WRITING:-}" == 1 ]]; then TPC_WRITING_PAD_STATUS_MAP=""; else TPC_WRITING_PAD_STATUS_MAP="--enableWritingPadStatusMap true"; fi

if ! workflow_has_parameter CALIB_LOCAL_INTEGRATED_AGGREGATOR; then
  if [[ $CALIB_TPC_IDC == 1 ]] && [[ $AGGREGATOR_TASKS == TPC_IDCBOTH_SAC || $AGGREGATOR_TASKS == ALL ]]; then
    add_W o2-tpc-idc-distribute "--crus ${crus} --timeframes ${nTFs} --output-lanes ${lanesFactorize} --send-precise-timestamp true --condition-tf-per-query ${nTFs} --n-TFs-buffer ${nBuffer}"
    add_W o2-tpc-idc-factorize "--n-TFs-buffer ${nBuffer} --input-lanes ${lanesFactorize} --crus ${crus} --timeframes ${nTFs} --nthreads-grouping ${threadFactorize} --nthreads-IDC-factorization ${threadFactorize} --sendOutputFFT true --enable-CCDB-output true --enablePadStatusMap true ${TPC_WRITING_PAD_STATUS_MAP} --use-precise-timestamp true $IDC_DELTA" "TPCIDCGroupParam.groupPadsSectorEdges=32211"
    add_W o2-tpc-idc-ft-aggregator "--rangeIDC 200 --inputLanes ${lanesFactorize} --nFourierCoeff 40 --nthreads 8"
  fi
  if [[ $CALIB_TPC_SAC == 1 ]] && [[ $AGGREGATOR_TASKS == TPC_IDCBOTH_SAC || $AGGREGATOR_TASKS == ALL ]]; then
    add_W o2-tpc-sac-distribute "--timeframes ${nTFs_SAC} --output-lanes 1 "
    add_W o2-tpc-sac-factorize "--timeframes ${nTFs_SAC} --nthreads-SAC-factorization 4 --input-lanes 1 --compression 2"
    add_W o2-tpc-idc-ft-aggregator "--rangeIDC 200 --nFourierCoeff 40 --process-SACs true --inputLanes 1"
  fi
fi

# Calo cal
# calibrations for AGGREGATOR_TASKS == CALO_TF
if [[ $AGGREGATOR_TASKS == CALO_TF || $AGGREGATOR_TASKS == ALL ]]; then
  # EMC
  EMCAL_CALIB_OPT=
  EMCAL_CALIB_CONFIG=
  if ! has_detector CTP; then
    EMCAL_CALIB_OPT+=" --no-rejectL0Trigger"
  fi
  [[ $EPNSYNCMODE == 1 ]] && EMCAL_CALIB_CONFIG+="EMCALCalibParams.filePathSave=/scratch/services/detector_tmp/emc_calib;"
  if [[ $CALIB_EMC_BADCHANNELCALIB == 1 ]]; then
    add_W o2-calibration-emcal-channel-calib-workflow "${EMCAL_CALIB_OPT} --calibType \"badchannels\"" "${EMCAL_CALIB_CONFIG}"
  fi
  if [[ $CALIB_EMC_TIMECALIB == 1 ]]; then
    add_W o2-calibration-emcal-channel-calib-workflow "${EMCAL_CALIB_OPT} --calibType \"time\"" "${EMCAL_CALIB_CONFIG}"
  fi

  # PHS
  if [[ $CALIB_PHS_ENERGYCALIB == 1 ]]; then
    add_W o2-phos-calib-workflow "--energy --phoscalib-output-dir $CALIB_DIR --phoscalib-meta-output-dir $EPN2EOS_METAFILES_DIR"
  fi
  if [[ $CALIB_PHS_BADMAPCALIB == 1 ]]; then
    add_W o2-phos-calib-workflow "--badmap --mode 0"
  fi
  if [[ $CALIB_PHS_TURNONCALIB == 1 ]]; then
    add_W o2-phos-calib-workflow "--turnon"
  fi
  if [[ $CALIB_PHS_RUNBYRUNCALIB == 1 ]]; then
    add_W o2-phos-calib-workflow "--runbyrun --phoscalib-output-dir $CALIB_DIR --phoscalib-meta-output-dir $EPN2EOS_METAFILES_DIR"
  fi
  if [[ $CALIB_PHS_L1PHASE == 1 ]]; then
    add_W o2-phos-calib-workflow "--l1phase"
  fi

  # CPV
  if [[ $CALIB_CPV_GAIN == 1 ]]; then
    add_W o2-calibration-cpv-calib-workflow "--gains"
  fi
fi

# Forward detectors
if [[ $AGGREGATOR_TASKS == FORWARD_TF || $AGGREGATOR_TASKS == ALL ]]; then
  # FT0
  if [[ $CALIB_FT0_TIMEOFFSET == 1 ]]; then
    add_W o2-calibration-ft0-time-offset-calib "--tf-per-slot $FT0_TIMEOFFSET_TF_PER_SLOT --max-delay 0" "FT0CalibParam.mNExtraSlots=0;FT0CalibParam.mRebinFactorPerChID[180]=4;FT0DigitFilterParam.mTrgBitsGood=${FT0_TIMEOFFSET_TRG_BITS};FT0DigitFilterParam.mTrgBitsToCheck=${FT0_TIMEOFFSET_TRG_BITS};"
  fi
fi

if [[ $AGGREGATOR_TASKS == FORWARD_SPORADIC || $AGGREGATOR_TASKS == ALL ]]; then
  # FT0
  if [[ $CALIB_FT0_INTEGRATEDCURR == 1 ]]; then
    add_W o2-ft0-merge-integrate-cluster-workflow "--tf-per-slot $INTEGRATEDCURR_TF_PER_SLOT"
  fi
  # FV0
  if [[ $CALIB_FV0_INTEGRATEDCURR == 1 ]]; then
    add_W o2-fv0-merge-integrate-cluster-workflow "--tf-per-slot $INTEGRATEDCURR_TF_PER_SLOT"
  fi
  # FDD
  if [[ $CALIB_FDD_INTEGRATEDCURR == 1 ]]; then
    add_W o2-fdd-merge-integrate-cluster-workflow "--tf-per-slot $INTEGRATEDCURR_TF_PER_SLOT"
  fi
  # ZDC
  if [[ $CALIB_ZDC_TDC == 1 ]]; then
    add_W o2-zdc-tdccalib-workflow "" "CalibParamZDC.outputDir=$CALIB_DIR;CalibParamZDC.metaFileDir=$EPN2EOS_METAFILES_DIR"
  fi
fi

if [[ "${GEN_TOPO_VERBOSE:-}" == "1" ]]; then
  # calibrations for AGGREGATOR_TASKS == CALO_SPORADIC
  if [[ $AGGREGATOR_TASKS == CALO_SPORADIC ]] || [[ $AGGREGATOR_TASKS == ALL ]]; then
    echo "AGGREGATOR_TASKS == CALO_SPORADIC not defined for the time being" 1>&2
  fi

  # calibrations for AGGREGATOR_TASKS == MUON_TF
  if [[ $AGGREGATOR_TASKS == MUON_TF ]] || [[ $AGGREGATOR_TASKS == ALL ]]; then
    echo "AGGREGATOR_TASKS == MUON_TF not defined for the time being" 1>&2
  fi

  # calibrations for AGGREGATOR_TASKS == MUON_SPORADIC
  if [[ $AGGREGATOR_TASKS == MUON_SPORADIC ]] || [[ $AGGREGATOR_TASKS == ALL ]]; then
    echo "AGGREGATOR_TASKS == MUON_SPORADIC not defined for the time being" 1>&2
  fi
fi

if [[ $CCDB_POPULATOR_UPLOAD_PATH != "none" ]] && [[ ! -z $WORKFLOW ]] && [[ $WORKFLOW != "echo '{}' | " ]]; then add_W o2-calibration-ccdb-populator-workflow "--ccdb-path $CCDB_POPULATOR_UPLOAD_PATH --environment \"DPL_DONT_DROP_OLD_TIMESLICE=1\" --sspec-min $CCDBPRO_SUBSPEC_MIN --sspec-max $CCDBPRO_SUBSPEC_MAX"; fi

if [[ $CCDB_DCS_POPULATOR_UPLOAD_PATH != "none" ]] && [[ ! -z $WORKFLOW ]] && [[ $WORKFLOW != "echo '{}' | " ]] && [[ $NEED_DCS_CCDB_POPULATOR != 0 ]]; then add_W o2-calibration-ccdb-populator-workflow "--ccdb-path $CCDB_DCS_POPULATOR_UPLOAD_PATH --environment \"DPL_DONT_DROP_OLD_TIMESLICE=1\" --sspec-min $CCDBDCS_SUBSPEC_MIN --sspec-max $CCDBDCS_SUBSPEC_MAX --name-extention dcs"; fi

if ! workflow_has_parameter CALIB_LOCAL_INTEGRATED_AGGREGATOR; then
  WORKFLOW+="o2-dpl-run $ARGS_ALL $GLOBALDPLOPT"
  [[ $WORKFLOWMODE != "print" ]] && WORKFLOW+=" --${WORKFLOWMODE} ${WORKFLOWMODE_FILE:-}"
  [[ $WORKFLOWMODE == "print" || "${PRINT_WORKFLOW:-}" == "1" ]] && echo "#Aggregator Workflow command:\n\n${WORKFLOW}\n" | sed -e "s/\\\\n/\n/g" -e"s/| */| \\\\\n/g" | eval cat $( [[ $WORKFLOWMODE == "dds" ]] && echo '1>&2')
  if [[ $WORKFLOWMODE != "print" ]] && [[ ! -z $WORKFLOW ]] && [[ $WORKFLOW != "echo '{}' | " ]]; then eval $WORKFLOW; else true; fi
fi
