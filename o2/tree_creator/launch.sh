LOGFILE="output.log"
CONF="-b --configuration json://configuration.json"
OUTPUT_DIR="--aod-writer-json output_director.json"
INPUT="--aod-file @input_data.txt"

o2-analysis-lf-cluster-studies-tree-creator $CONF|\

    # converters
    # o2-analysis-tracks-extra-converter $CONF|
    # o2-analysis-mc-converter $CONF|
    # o2-analysis-bc-converter $CONF|

    # standard wagons
    o2-analysis-timestamp $CONF|\
    o2-analysis-event-selection $CONF|\
    o2-analysis-track-propagation $CONF|\
    o2-analysis-trackselection $CONF|\
    o2-analysis-multiplicity-table $CONF|\
    o2-analysis-ft0-corrected-table $CONF|\
    o2-analysis-lf-lambdakzerobuilder $CONF|\
    o2-analysis-pid-tpc-base $CONF|\
    o2-analysis-pid-tpc $CONF|\
    o2-analysis-pid-tof $CONF|\
    o2-analysis-pid-tof-base $CONF|\
    o2-analysis-pid-tof-full $CONF $INPUT $OUTPUT_DIR > $LOGFILE


# report the status of the workflow
rc=$?
if [ $rc -eq 0 ]; then
    echo "Workflow finished successfully"
else
    echo "Error: Workflow failed with status $rc"
    echo "Check the log file for more details: $LOGFILE"
    exit $rc
fi

    