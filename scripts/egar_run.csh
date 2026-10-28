#!/bin/tcsh

cd /work/dfrest/develop/users/tarhini1/AGxAG/ImagineAgents
source setup.csh
cd /work/dfrest/develop/users/tarhini1/AGxAG/ImagineAgents/EGAR-DRC

# set run_arguments = ( --jobs 128 )
# if ($#argv > 0) set run_arguments = ( $argv:q )

/work/dfrest/develop/users/tarhini1/AGxAG/.venv/bin/python -m egar_drc.runtime.runner \
    --flow egar-drc \
    --model-config /work/dfrest/develop/users/tarhini1/AGxAG/ImagineAgents/config/egar_drc_20b.yaml \
    --repository /prj/ai4drc/users/tarhini1/Rule2DRC \
    --klayout-bin /sw/freetools/klayout/0.30.7/rh80_64/bin/klayout \
    --output-dir /prj/ai4drc/common/implicit_results/phd_data/rule2drc_flow_full_corr/agentic_flow/egar_full_120b_prompt_rep1 \
    --jobs 128
    # $run_arguments:q