#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys
#import imp
import glob
import numpy
from shutil import copyfile

parser = argparse.ArgumentParser(
    description='schedules climada runs for different parameter combinations')
# parser.add_argument(
#     '--parameters', type=str, default="parameters.py",
#     help='parameters file')
parser.add_argument(
    '--dry', action="store_true",
    help='dry run (do not run Climada)')
parser.add_argument(
    '--shared', action="store_true",
    help='share nodes on cluster')
parser.add_argument(
    '--notify', action="store_true",
    help='notify per mail when done')
parser.add_argument(
    '--minutes', type=int, default=0,
    help='maximal minutes to run on cluster (< 60)')
parser.add_argument(
    '--hours', type=int, default=24,
    help='maximal hours to run on cluster (168=week, 720=month)')
parser.add_argument(
    '--threads', type=int, default=16,
    help='maximal number of threads on cluster (<= 16)')
parser.add_argument(
    '--mem_per_cpu', type=int, default=3584,
    help='number of memory per CPU (3584 is MaxMemPerCPU on cluster)')
parser.add_argument(
    '--largemem', action="store_true",
    help='use ram_gpu partition')
parser.add_argument(
    '--verbose', action="store_true",
    help='be verbose')

args = parser.parse_args()

indices = []
sys.dont_write_bytecode = True

#RF_MODEL = numpy.arange(1980,2017)

SCENARIO = ['rcp26',
            'rcp60']

CL_MODEL = [#'gfdl-esm2m',
            #'hadgem2-es',
            #'ipsl-cm5a-lr',
            #'miroc5'
            'princeton',
            'gswp3',
            'watch',
            'wfdei'
            ]

RF_MODEL = [
            'clm40',
            'matsiro',
            'jules-w1',
            'jules-b1',
            'orchidee',
            'vic',
            'dbh',
            'h08',
            'lpjml',
            'mpi-hm',
            'pcr-globwb',
            'watergap2']

def schedule_run(run_nb,flag,RF_model,CL_model, Socmode, SM_mode):
    if not flag:
        run_label = "run%s" % run_nb
        if os.path.exists(run_label):
        #    run_id += 1
            return
        os.mkdir(run_label)
 #       desc = run_description()
 #       f = open("%s/parameters.txt" % run_label, 'w')
 #       f.write(desc)
 #       f.close()
 #       run_index.write(run_description_csv(run_label))
 #       run_index.write("\n")
        # with open("%s/settings.yml" % run_label, 'w') as f:
        #     f.write(pyaml.dump(settings_yml))
        #     for nc in glob.glob('*.nc'):
        #         copyfile(nc,"%s/%s" % (run_label,nc))
        #run_id += 1
    else:
        run_label = "."
    if args.dry:
        return
    else:
        if (int(args.hours) <= 24):
            _class = "short"
        elif (int(args.hours) <= 24 * 7):
            _class = "medium"
        else:
            _class = "long"

        run_params = {
            "job_name": "%s/%s" % (os.path.basename(os.getcwd()), run_label),
            "minutes": args.minutes,
            "hours": args.hours,
            "class": _class,
            "initialdir": run_label,
            "node_usage": "share" if args.shared else "exclusive",
            "notification": "END,FAIL,TIME_LIMIT" if args.notify else "FAIL,TIME_LIMIT",
            "comment": "%s/%s" % (os.getcwd(), run_label),
            "environment": "ALL",
            "executable": 'schedule_sim.py',
            "options": "--RF_model %s --CL_model %s --Socmode %s --SM_mode %s"%(RF_model, CL_model, Socmode, SM_mode),
            "num_threads": args.threads,
            "mem_per_cpu": args.mem_per_cpu if not args.largemem else 15360,   # if mem_per_cpu is larger than MaxMemPerCPU then num_threads is reduced
            "other": "#SBATCH --partition=ram_gpu" if args.largemem else ""
        }

        cmd = """echo "#!/bin/sh
#SBATCH --job-name=\\\"%(job_name)s\\\"
#SBATCH --comment=\\\"%(comment)s\\\"
#SBATCH --time=%(hours)02d:%(minutes)02d:00
#SBATCH --qos=%(class)s
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --export=%(environment)s
#SBATCH --mail-type=%(notification)s
#SBATCH --%(node_usage)s
#SBATCH --account=ebm        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(num_threads)1d
#SBATCH --mem-per-cpu=%(mem_per_cpu)1d
#SBATCH --workdir=%(initialdir)s
%(other)s
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=%(num_threads)1d
source activate climada_env        
ulimit -c unlimited
%(executable)s %(options)s
" | sbatch -Q""" % run_params

        if args.verbose:
            print(cmd)

        os.system(cmd)
        #run_cnt += 1

num = 1
num *= len(RF_MODEL)
num *= len(CL_MODEL)

single = True if num == 1 else False
if num > 1:
    print("Number of runs to be scheduled: %s" % num)
    sys.stdout.write('Run? y/N : ')
    if sys.version_info >= (3, 0):
        if input() != "y":
            exit("Aborted")
    else:
        if raw_input() != "y":
            exit("Aborted")

enum = 1
for rf_model in RF_MODEL:
    for cl_model in CL_MODEL:
        schedule_run(run_nb=enum,flag=single,RF_model=rf_model,CL_model=cl_model, Socmode = 'nosoc', SM_mode ='exact')
        enum += 1
if num > 1:
    print("Scheduled %s runs" % num)

# def set_in_yml(paths, value):
#     global yml_nodes
#     for p in paths:
#         node = yml_nodes
#         nodes = p.split(".")
#         for n in nodes[:-1]:
#             try:
#                 n = int(n)
#             except ValueError:
#                 if not n in node:
#                     exit("Path '%s' not found!" % p)
#             node = node[n]
#         n = nodes[-1]
#         if not n in node:
#             exit("Path '%s' not found!" % p)
#         node[n] = value


# def next_step():
#     for i, ind in enumerate(indices):
#         indices[i] += 1
#         if indices[i] < len(parameters[i]["values"]):
#             set_in_yml(
#                 parameters[i]["paths"], parameters[i]["values"][indices[i]])
#             return True
#         else:
#             indices[i] = 0
#             set_in_yml(
#                 parameters[i]["paths"], parameters[i]["values"][indices[i]])



# if os.path.exists(args.parameters):
#     parameters = imp.load_source("parameters", args.parameters).parameters
#     single = False
# else:
#     single = True
# run_cnt = 0
# def run_description():
#     res = ""
#     for i, ind in enumerate(indices):
#         res += "%s = %s\n" % (parameters[i]
#                               ["name"], parameters[i]["values"][ind])
#     return res


# def run_description_csv(run_label):
#     res = "\"%s\"" % run_label
#     for i, ind in enumerate(indices):
#         res += ",\"%s\"" % parameters[i]["values"][ind]
#     return res
