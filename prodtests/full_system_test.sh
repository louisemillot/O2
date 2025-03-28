#!/bin/bash
#
# A workflow performing a full system test:
# - simulation of digits
# - creation of raw data
# - reconstruction of raw data
#
# Note that this might require a production server to run.
#
# This script can use additional binary objects which can be optionally provided:
# - matbud.root
#
# authors: D. Rohr / S. Wenzel

if [ "0$O2_ROOT" == "0" ] || [ "0$AEGIS_ROOT" == "0" ]; then
  echo Missing O2sim environment
  exit 1
fi

if [[ $DPL_CONDITION_BACKEND != "http://o2-ccdb.internal" && $DPL_CONDITION_BACKEND != "http://localhost:8084" && $DPL_CONDITION_BACKEND != "http://127.0.0.1:8084" ]]; then
  alien-token-info >& /dev/null
  RETVAL=$?
  if [[ $RETVAL != 0 ]]; then
    echo "command alien-token-init had nonzero exit code $RETVAL" 1>&2
    echo "FATAL: No alien token present" 1>&2
    exit 1
  fi
fi

# include jobutils, which notably brings
# --> the taskwrapper as a simple control and monitoring tool
#     (look inside the jobutils.sh file for documentation)
# --> utilities to query CPU count
. ${O2_ROOT}/share/scripts/jobutils.sh

# make sure that correct format will be used irrespecive of the locale
export LC_NUMERIC=C
export LC_ALL=C

BEAMTYPE=${BEAMTYPE:-PbPb}
NEvents=${NEvents:-10} #550 for full TF (the number of PbPb events)
NEventsQED=${NEventsQED:-1000} #35000 for full TF
NCPUS=$(getNumberOfPhysicalCPUCores)
echo "Found ${NCPUS} physical CPU cores"
NJOBS=${NJOBS:-"${NCPUS}"}
SHMSIZE=${SHMSIZE:-8000000000} # Size of shared memory for messages (use 128 GB for 550 event full TF)
#TPCTRACKERSCRATCHMEMORY=${TPCTRACKERSCRATCHMEMORY:-8000000000} # Not needed by default
ENABLE_GPU_TEST=${ENABLE_GPU_TEST:-0} # Run the full system test also on the GPU
GPUMEMSIZE=${GPUMEMSIZE:-6000000000} # Size of GPU memory to use in case ENABBLE_GPU_TEST=1
NTIMEFRAMES=${NTIMEFRAMES:-1} # Number of time frames to process
TFDELAY=${TFDELAY:-100} # Delay in seconds between publishing time frames
[[ -z ${NOMCLABELS+x} ]] && NOMCLABELS="--disable-mc"
O2SIMSEED=${O2SIMSEED:-0}
SPLITTRDDIGI=${SPLITTRDDIGI:-1}
DIGITDOWNSCALINGTRD=${DIGITDOWNSCALINGTRD:-1000}
NHBPERTF=${NHBPERTF:-128}
RUNFIRSTORBIT=${RUNFIRSTORBIT:-0}
FIRSTSAMPLEDORBIT=${FIRSTSAMPLEDORBIT:-0}
OBLIGATORYSOR=${OBLIGATORYSOR:-false}
FST_TPC_ZSVERSION=${FST_TPC_ZSVERSION:-4}
TPC_SLOW_REALISITC_FULL_SIM=${TPC_SLOW_REALISITC_FULL_SIM:-0}
FST_BFIELD="${FST_BFIELD:-}ccdb"
if [[ $BEAMTYPE == "PbPb" ]]; then
  FST_GENERATOR=${FST_GENERATOR:-pythia8hi}
  FST_COLRATE=${FST_COLRATE:-50000}
  RUNNUMBER=310000 # a default un-anchored Pb-Pb run number
else
  FST_GENERATOR=${FST_GENERATOR:-pythia8pp}
  FST_COLRATE=${FST_COLRATE:-400000}
  RUNNUMBER=303000 # a default un-anchored pp run number
fi
FST_MC_ENGINE=${FST_MC_ENGINE:-TGeant4}
FST_EMBEDDING_CONFIG=${FST_EMBEDDING_CONFIG:-GeneratorPythia8.config=$O2_ROOT/prodtests/full-system-test/pythia8.cfg}
DO_EMBEDDING=${DO_EMBEDDING:-0}
if [[ $DO_EMBEDDING == 0 ]]; then
  SIM_SOURCES="o2sim"
else
  SIM_SOURCES="sig,o2sim"
fi

[[ "$FIRSTSAMPLEDORBIT" -lt "$RUNFIRSTORBIT" ]] && FIRSTSAMPLEDORBIT=$RUNFIRSTORBIT

# allow skipping
JOBUTILS_SKIPDONE=ON
# potentially enable memory monitoring (independent on whether DPL or not)
# JOBUTILS_MONITORMEM=ON
# CPU monitoring JOBUTILS_MONITORCPU=ON

# prepare some metrics file for the monitoring system
METRICFILE=metrics.dat
CONFIG="full_system_test_N${NEvents}"
HOST=`hostname`

# include header information such as tested alidist tag and O2 tag
TAG="conf=${CONFIG},host=${HOST}${ALIDISTCOMMIT:+,alidist=$ALIDISTCOMMIT}${O2COMMIT:+,o2=$O2COMMIT}"
echo "versions,${TAG} alidist=\"${ALIDISTCOMMIT}\",O2=\"${O2COMMIT}\" " > ${METRICFILE}

GLOBALDPLOPT="-b" # --monitoring-backend no-op:// is currently removed due to https://alice.its.cern.ch/jira/browse/O2-1887

HBFUTILPARAMS="HBFUtils.nHBFPerTF=${NHBPERTF};HBFUtils.orbitFirst=${RUNFIRSTORBIT};HBFUtils.orbitFirstSampled=${FIRSTSAMPLEDORBIT};HBFUtils.obligatorySOR=${OBLIGATORYSOR};HBFUtils.runNumber=${RUNNUMBER};"
[[ "0$ALLOW_MULTIPLE_TF" != "01" || "0$ALLOW_MULTIPLE_TF_N" != "01" ]] && HBFUTILPARAMS+=";HBFUtils.maxNOrbits=$((${FIRSTSAMPLEDORBIT} + ${ALLOW_MULTIPLE_TF_N:-1} * ${NHBPERTF}));"

ulimit -n 4096 # Make sure we can open sufficiently many files
[[ $? == 0 ]] || (echo Failed setting ulimit && exit 1)

if [[ $BEAMTYPE == "PbPb" && -z $FST_QED ]]; then
  FST_QED=1
fi
DIGIQED=
SIMOPTKEY="Diamond.width[2]=6.;"
if [[ $FST_QED == 1 ]]; then
  mkdir -p qed
  cd qed
  PbPbXSec="8."
  taskwrapper qedsim.log o2-sim ${FST_BFIELD+--field=}${FST_BFIELD} --seed $O2SIMSEED -j $NJOBS -n $NEventsQED -m PIPE ITS MFT FT0 FV0 FDD -g extgen -e ${FST_MC_ENGINE} --configKeyValues '"GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/QEDLoader.C;QEDGenParam.yMin=-7;QEDGenParam.yMax=7;QEDGenParam.ptMin=0.001;QEDGenParam.ptMax=1.;${SIMOPTKEY}"' --run ${RUNNUMBER}
  QED2HAD=$(awk "BEGIN {printf \"%.2f\",`grep xSectionQED qedgenparam.ini | cut -d'=' -f 2`/$PbPbXSec}")
  echo "Obtained ratio of QED to hadronic x-sections = $QED2HAD" >> qedsim.log
  cd ..
  DIGIQED="--simPrefixQED qed/o2sim --qed-x-section-ratio ${QED2HAD}"
fi

DIGITOPT=
DIGITOPTKEYTRD="TRDSimParams.digithreads=${NJOBS};"
DIGITOPTKEY=${HBFUTILPARAMS}
[[ ! -z $ITS_STROBE ]] && DIGITOPTKEY+="ITSAlpideParam.roFrameLengthInBC=$ITS_STROBE;"
[[ ! -z $MFT_STROBE ]] && DIGITOPTKEY+="MFTAlpideParam.roFrameLengthInBC=$MFT_STROBE;"
if [ $SPLITTRDDIGI == "1" ]; then
  DIGITOPT+=" --skipDet TRD"
  DIGITOPTKEYTRD+=${HBFUTILPARAMS}
else
  DIGITOPT+=" --trd-digit-downscaling ${DIGITDOWNSCALINGTRD}"
  DIGITOPTKEY+=$DIGITOPTKEYTRD
fi

if [[ $TPC_SLOW_REALISITC_FULL_SIM == 1 ]]; then
  SIMOPTKEY+="G4.physicsmode=3;SimCutParams.lowneut=true;TPCEleParam.doCommonModePerPad=0;TPCEleParam.doIonTailPerPad=1;TPCEleParam.commonModeCoupling=0;TPCEleParam.doNoiseEmptyPads=1;TPCEleParam.doSaturationTail=0;TPCDetParam.TPCRecoWindowSim=10;"
  DIGITOPTKEY+="TPCEleParam.doCommonModePerPad=0;TPCEleParam.doIonTailPerPad=1;TPCEleParam.commonModeCoupling=0;TPCEleParam.doNoiseEmptyPads=1;TPCEleParam.doSaturationTail=0;TPCDetParam.TPCRecoWindowSim=10;"
fi

taskwrapper sim.log o2-sim ${FST_BFIELD+--field=}${FST_BFIELD} --seed $O2SIMSEED -n $NEvents --configKeyValues "\"$SIMOPTKEY\"" -g ${FST_GENERATOR} -e ${FST_MC_ENGINE} -j $NJOBS --run ${RUNNUMBER} -o o2sim
if [[ $DO_EMBEDDING == 1 ]]; then
  taskwrapper embed.log o2-sim ${FST_BFIELD+--field=}${FST_BFIELD} -j $NJOBS --run ${RUNNUMBER} -n $NEvents -g pythia8pp -e ${FST_MC_ENGINE} -o sig --configKeyValues ${FST_EMBEDDING_CONFIG} --embedIntoFile o2sim_Kine.root
fi
taskwrapper digi.log o2-sim-digitizer-workflow -n $NEvents ${DIGIQED} ${NOMCLABELS} --sims ${SIM_SOURCES} --tpc-lanes $((NJOBS < 36 ? NJOBS : 36)) --shm-segment-size $SHMSIZE ${GLOBALDPLOPT} ${DIGITOPT} --configKeyValues "\"${DIGITOPTKEY}\"" --interactionRate $FST_COLRATE --early-forward-policy always
[[ $SPLITTRDDIGI == "1" ]] && taskwrapper digiTRD.log o2-sim-digitizer-workflow -n $NEvents ${NOMCLABELS} --sims ${SIM_SOURCES} --onlyDet TRD --trd-digit-downscaling ${DIGITDOWNSCALINGTRD} --shm-segment-size $SHMSIZE ${GLOBALDPLOPT} --incontext collisioncontext.root --configKeyValues "\"${DIGITOPTKEYTRD}\"" --early-forward-policy always
touch digiTRD.log_done

if [[ "0$GENERATE_ITSMFT_DICTIONARIES" == "01" ]]; then
  taskwrapper itsmftdict1.log o2-its-reco-workflow --trackerCA --disable-mc --configKeyValues '"fastMultConfig.cutMultClusLow=30000;fastMultConfig.cutMultClusHigh=2000000;fastMultConfig.cutMultVtxHigh=500;"'
  cp ~/alice/O2/Detectors/ITSMFT/ITS/macros/test/CreateDictionaries.C .
  taskwrapper itsmftdict2.log root -b -q CreateDictionaries.C++
  rm -f CreateDictionaries_C* CreateDictionaries.C
  taskwrapper itsmftdict3.log o2-mft-reco-workflow --disable-mc
  cp ~/alice/O2/Detectors/ITSMFT/MFT/macros/test/CreateDictionaries.C .
  taskwrapper itsmftdict4.log root -b -q CreateDictionaries.C++
  rm -f CreateDictionaries_C* CreateDictionaries.C
fi

mkdir -p raw
taskwrapper itsraw.log o2-its-digi2raw --file-for cruendpoint -o raw/ITS
taskwrapper mftraw.log o2-mft-digi2raw --file-for cruendpoint -o raw/MFT
taskwrapper ft0raw.log o2-ft0-digi2raw --file-for cruendpoint -o raw/FT0
taskwrapper fv0raw.log o2-fv0-digi2raw --file-for cruendpoint -o raw/FV0
taskwrapper fddraw.log o2-fdd-digit2raw --file-for cruendpoint -o raw/FDD
taskwrapper tpcraw.log o2-tpc-digits-to-rawzs --zs-version ${FST_TPC_ZSVERSION} --file-for cruendpoint -i tpcdigits.root -o raw/TPC
taskwrapper tofraw.log o2-tof-reco-workflow ${GLOBALDPLOPT} --file-for cruendpoint --output-type raw --tof-raw-outdir raw/TOF
taskwrapper midraw.log o2-mid-digits-to-raw-workflow ${GLOBALDPLOPT} --mid-raw-outdir raw/MID --file-for cruendpoint
taskwrapper mchraw.log o2-mch-digits-to-raw --input-file mchdigits.root --output-dir raw/MCH --file-for cruendpoint
taskwrapper emcraw.log o2-emcal-rawcreator --file-for link -o raw/EMC
taskwrapper phsraw.log o2-phos-digi2raw --file-for link -o raw/PHS
taskwrapper cpvraw.log o2-cpv-digi2raw --file-for cruendpoint -o raw/CPV
taskwrapper zdcraw.log o2-zdc-digi2raw --file-for cruendpoint -o raw/ZDC
taskwrapper hmpraw.log o2-hmpid-digits-to-raw-workflow --file-for crorcendpoint --outdir raw/HMP
taskwrapper trdraw.log o2-trd-trap2raw -o raw/TRD --file-for cruendpoint
taskwrapper ctpraw.log o2-ctp-digi2raw -o raw/CTP --file-for cruendpoint

CHECK_DETECTORS_RAW="ITS MFT FT0 FV0 FDD TPC TOF MID MCH CPV ZDC TRD CTP"
if [[ $BEAMTYPE == "PbPb" ]] && [ $NEvents -ge 5 ] ; then
  CHECK_DETECTORS_RAW+=" EMC PHS HMP"
fi
for i in $CHECK_DETECTORS_RAW; do
  if [[ `ls -l raw/$i/*.raw | awk '{print $5}' | grep -v "^0\$" | wc -l` == "0" ]]; then
    echo "ERROR: Full system test did generate no raw data for $i"
    exit 1
  fi
done

cat raw/*/*.cfg > rawAll.cfg

if [[ "0$DISABLE_PROCESSING" == "01" ]]; then
  echo "Skipping the processing part of the full system test"
  exit 0
fi

# We run the workflow in both CPU-only and With-GPU mode
STAGES="NOGPU"
if [[ $ENABLE_GPU_TEST != "0" ]]; then
  STAGES+=" WITHGPU"
fi
STAGES+=" ASYNC"

if [[ ${RANS_OPT:-} =~ (--ans-version +)(compat) ]] ; then
  # Give a possibility to run the FST with external existing dictionary (i.e. with CREATECTFDICT=0 full_system_test.sh)
  # In order to use CCDB dictionaries, pass CTFDICTFILE=ccdb CREATECTFDICT=0
  [[ ! -z "$CREATECTFDICT" ]] && SYNCMODEDOCTFDICT="$CREATECTFDICT" || SYNCMODEDOCTFDICT=1

  # this is default local tree-based CTF dictionary file
  [[ -z "$CTFDICTFILE" ]] && CTFDICTFILE="ctf_dictionary.root"

  # if dictionary creation is requested, the encoders should not use any external dictionary, neither the local one (--ctf-dict <file>) nor from the CCDB (empty or --ctf-dict "ccdb")
  [[ "$SYNCMODEDOCTFDICT" = "1" ]] && USECTFDICTFILE="none" || USECTFDICTFILE="$CTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_itsmft_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_ft0_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_fv0_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_mid_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_mch_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_phos_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_cpv_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_emcal_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_zdc_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_fdd_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_hmpid_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_tof_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_trd_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_tpc_reco_workflow+="--ctf-dict $USECTFDICTFILE"
  export ARGS_EXTRA_PROCESS_o2_ctp_entropy_encoder_workflow+="--ctf-dict $USECTFDICTFILE"

  # for decoding we use either just produced or externally provided common local file
  export ARGS_EXTRA_PROCESS_o2_ctf_reader_workflow+="--ctf-dict $CTFDICTFILE"
fi
export CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow+="GPU_global.overrideNHbfPerTF=$NHBPERTF;"

for STAGE in $STAGES; do
  logfile=reco_${STAGE}.log

  ARGS_ALL="--session default"
  DICTCREATION=""
  export WORKFLOW_PARAMETERS=
  if [[ "$STAGE" = "WITHGPU" ]]; then
    export CREATECTFDICT=0
    export GPUTYPE=CUDA
    export HOSTMEMSIZE=1000000000
    export SYNCMODE=1
    export CTFINPUT=0
    # enabling SECVTX
    export WORKFLOW_EXTRA_PROCESSING_STEPS+="MATCH_SECVTX"
  elif [[ "$STAGE" = "ASYNC" ]]; then
    export CREATECTFDICT=0
    export GPUTYPE=CPU
    export SYNCMODE=0
    export HOSTMEMSIZE=$TPCTRACKERSCRATCHMEMORY
    export CTFINPUT=1
    # the following line is needed in case the SECTVX was enabled in the SYNC; in this case, it'd have the options:
    # export ARGS_EXTRA_PROCESS_o2_secondary_vertexing_workflow='--disable-cascade-finder --disable-3body-finder --disable-strangeness-tracker'
    unset ARGS_EXTRA_PROCESS_o2_secondary_vertexing_workflow
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},AOD"
  else
    export CREATECTFDICT=$SYNCMODEDOCTFDICT
    export GPUTYPE=CPU
    export SYNCMODE=1
    export HOSTMEMSIZE=$TPCTRACKERSCRATCHMEMORY
    export CTFINPUT=0
    export WORKFLOW_PARAMETERS="${WORKFLOW_PARAMETERS},CALIB,CTF,EVENT_DISPLAY,${FST_SYNC_EXTRA_WORKFLOW_PARAMETERS}"
    # enabling SECVTX
    export WORKFLOW_EXTRA_PROCESSING_STEPS+="MATCH_SECVTX"
    # temporarily enable ZDC reconstruction for calibration validations
    export WORKFLOW_EXTRA_PROCESSING_STEPS+=",ZDC_RECO"
    unset JOBUTILS_JOB_SKIPCREATEDONE
  fi
  export SHMSIZE
  export NTIMEFRAMES
  export TFDELAY
  export GLOBALDPLOPT
  export GPUMEMSIZE

  # do not prescale ITS reconstruction in PbPb unless explicitly requested
  if [[ -z ${FST_PRESCALE_ITS:-} ]] ; then
    : ${CUT_RANDOM_FRACTION_ITS:=-1}
    : ${CUT_MULT_MIN_ITS:=-1}
    : ${CUT_MULT_MAX_ITS:=-1}
    : ${CUT_MULT_VTX_ITS:=-1}
    : ${CUT_TRACKLETSPERCLUSTER_MAX_ITS:=100}
    : ${CUT_CELLSPERCLUSTER_MAX_ITS:=100}
    export CUT_TRACKLETSPERCLUSTER_MAX_ITS
    export CUT_CELLSPERCLUSTER_MAX_ITS
    export CUT_RANDOM_FRACTION_ITS
    export CUT_MULT_MIN_ITS
    export CUT_MULT_MAX_ITS
    export CUT_MULT_VTX_ITS
  fi

  taskwrapper ${logfile} "$O2_ROOT/prodtests/full-system-test/dpl-workflow.sh"

  # --- record interesting metrics to monitor ----
  # boolean flag indicating if workflow completed successfully at all
  RC=$?
  SUCCESS=0
  [[ -f "${logfile}_done" ]] && [[ "$RC" = 0 ]] && SUCCESS=1
  echo "success_${STAGE},${TAG} value=${SUCCESS}" >> ${METRICFILE}

  if [[ "${SUCCESS}" = "1" ]]; then
    # runtime
    walltime=`grep "#walltime" ${logfile}_time | awk '//{print $2}'`
    echo "walltime_${STAGE},${TAG} value=${walltime}" >> ${METRICFILE}

    # GPU reconstruction (also in CPU version) processing time
    gpurecotime=`grep -a "gpu-reconstruction" ${logfile} | grep -e "Total Wall Time:" | awk '//{printf "%f", $6/1000000}'`
    echo "gpurecotime_${STAGE},${TAG} value=${gpurecotime}" >> ${METRICFILE}

    # memory
    maxmem=`awk '/PROCESS MAX MEM/{print $5}' ${logfile}` # in MB
    avgmem=`awk '/PROCESS AVG MEM/{print $5}' ${logfile}` # in MB
    echo "maxmem_${STAGE},${TAG} value=${maxmem}" >> ${METRICFILE}
    echo "avgmem_${STAGE},${TAG} value=${avgmem}" >> ${METRICFILE}

    # some physics quantities
    tpctracks=`grep -a "gpu-reconstruction" ${logfile} | grep -e "found.*track" | awk '//{print $4}'`
    echo "tpctracks_${STAGE},${TAG} value=${tpctracks}" >> ${METRICFILE}
    tpcclusters=`grep -a -e "Event has.*TPC Clusters" ${logfile} | awk '//{print $5}'`
    echo "tpcclusters_${STAGE},${TAG} value=${tpcclusters}" >> ${METRICFILE}
  fi
done
