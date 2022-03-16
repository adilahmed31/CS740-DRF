# Fair Resource Allocation

This projects implements three resource sharing algorithms as mentioned in the Dominant Resource Fairness paper, and allows for comparison between the various approaches.

The code takes as input a csv file of tasks in the below format.

```
Resource_name1,Resource_name2,...,Resource_nameN
Capacity1, Capacity2,...,CapacityN
User1_Demand1,User1_Demand2....,User1_DemandN
User2_Demand1,User2_Demand2....,User2_DemandN
.
.
.
UserX_Demand1,UserX_Demand2....,UserX_DemandN
```
An example `tasks.csv` file is included. Optionally, one can provide a `weights.csv` file in the format provided.

## Usage:

```
usage: allocator.py [-h] [-i INPUT] [-a ALGO] [-g] [-w [WEIGHTS]]

Run different algorithms for resource allocation and compare results

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Provide the path of input CSV file
  -a ALGO, --algo ALGO  Specify resource sharing algorithm to use (int value). 1. DRF (default) 2.Asset Fairness 3.CEEI
  -g, --graph           Specify if graph should be generated for the allocation
  -w [WEIGHTS], --weights [WEIGHTS]
                        [Optional] Provide the path of weights CSV file
```

Example usage:
`python allocator.py -i tasks.csv -a 1 -g`

To use default values, simply run `python allocator.py`

All graphs are present in the `charts` folder. All test workloads are present in the `tests` folder.