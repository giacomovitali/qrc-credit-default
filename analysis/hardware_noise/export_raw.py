"""Export the raw Aquila records and the executed programs without device or account identifiers.

Reads the experiment directories of the hardware runs (data/experiments/QRC_12_emu_false_<run>/) and,
for every datapoint, the run whose records the features use (the `source` column of the records
index written by export_bits.jl). Per encoding it writes, to results/hardware_noise/raw/:

  aquila_<enc>_shots.npz          pre  (uint8, shots x traps)  traps filled before the pulse
                                  post (uint8, shots x traps)  measured state after the pulse
                                  completed (bool, shots)      shot status reported "Completed"
  aquila_<enc>_shots_index.csv    id, readout (1-5 = 0.5-2.5 µs), run, task (neutral index), offset,
                                  n_shots (rows offset .. offset + n_shots - 1 of the arrays), requested_shots,
                                  resubmitted (the task returned no shots and was run again on its own)
  aquila_<enc>_programs.jsonl.gz  one line per task: id, readout, run, task, and the executed program
                                  (nshots, lattice, effective_hamiltonian) as submitted

Task ARNs, device ARNs, job identifiers and file names of the provider are not written; the task index
is assigned here in (id, readout) order. Failed shots are kept with status False and zero sequences.

    python analysis/hardware_noise/export_raw.py

Run from the repository root (conda env qrc_class). Needs the experiment directories, which are not
part of the data deposit; the deposit holds this script's output.
"""
import gzip
import json

import numpy as np
import pandas as pd

from common import BITS, OUT, ROOT

EXP = ROOT / "data" / "experiments"
PREFIX = {"detuning": "aquila_det", "position": "aquila_pos"}
TRAPS = {"detuning": 72, "position": 78}       # 6 copies x 12 or 13 atoms
STEP_LABELS = ["0.0", "0.5", "1.0", "1.5", "2.0"]   # file labels of readouts 1-5


def readout_of(program):
    """Readout index 1-5 from the program duration (run time = T - 0.12 µs, T = 0.5 ... 2.5 µs)."""
    t_end = program["effective_hamiltonian"]["rydberg"]["rabi_frequency_amplitude"]
    t_end = (t_end.get("global") or t_end.get("global_"))["times"][-1] * 1e6
    return int(round(t_end / 0.5))


def tasks_of(exp_dir, id_):
    """[(program, requested shots or None, shot_outputs, resubmitted)] in readout order. A Braket task that
    returned no shots was run again on its own; its results are in results_<id>_step_<label>.json."""
    d = json.load(open(exp_dir / "results" / f"task_{id_}" / "raw_data" / f"results_{id_}.json"))
    if "tasks" in d:                               # direct access: programs stored at submission
        programs = [json.load(open(f)) for f in (exp_dir / "tasks" / "raw_data").glob(f"{id_}_*.json")]
        programs = {readout_of(p): p for p in programs}
        assert sorted(programs) == [1, 2, 3, 4, 5], (exp_dir.name, id_, sorted(programs))
        return [(programs[r], None, t["shot_outputs"], False) for r, t in enumerate(d["tasks"], start=1)]
    out = []
    for r, (_, t) in enumerate(d["bloqade.task.batch.RemoteBatch"]["tasks"], start=1):
        t = t["bloqade.task.braket.BraketTask"]
        resubmitted = False
        if not (t.get("task_result_ir") or {}).get("shot_outputs"):     # failed task, run again on its own
            f = exp_dir / "results" / f"task_{id_}" / "raw_data" / f"results_{id_}_step_{STEP_LABELS[r - 1]}.json"
            if f.exists():
                t, resubmitted = json.load(open(f))["bloqade.task.braket.BraketTask"], True
        shots = (t.get("task_result_ir") or {}).get("shot_outputs") or []
        out.append((t["task_ir"], t["task_ir"]["nshots"], shots, resubmitted))
    return out


def main():
    dest = OUT / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    for enc, prefix in PREFIX.items():
        sources = pd.read_csv(BITS / f"{prefix}_index.csv").groupby("id").source.first()
        pre, post, done, index = [], [], [], []
        offset, task = 0, 0
        with gzip.open(dest / f"{prefix}_programs.jsonl.gz", "wt") as programs:
            for id_, source in sources.items():
                for r, (program, requested, shots, resubmitted) in enumerate(tasks_of(EXP / source, id_), start=1):
                    assert readout_of(program) == r, (source, id_, r)
                    task += 1
                    run = source.replace("QRC_12_emu_false_", "")
                    for s in shots:
                        ok = s["shot_status"] == "Completed" and len(s.get("pre_sequence") or []) == TRAPS[enc]
                        pre.append(s["pre_sequence"] if ok else [0] * TRAPS[enc])
                        post.append(s["post_sequence"] if ok else [0] * TRAPS[enc])
                        done.append(ok)
                    index.append(dict(id=id_, readout=r, run=run, task=task, offset=offset, n_shots=len(shots),
                                      requested_shots=requested, resubmitted=resubmitted))
                    offset += len(shots)
                    programs.write(json.dumps(dict(id=int(id_), readout=r, run=run, task=task,
                                                   program={k: program[k] for k in ("nshots", "lattice", "effective_hamiltonian")})) + "\n")
        np.savez_compressed(dest / f"{prefix}_shots.npz", pre=np.array(pre, np.uint8), post=np.array(post, np.uint8),
                            completed=np.array(done, bool))
        pd.DataFrame(index).to_csv(dest / f"{prefix}_shots_index.csv", index=False)
        print(f"{enc}: {len(sources)} datapoints, {task} tasks, {offset} shots ({sum(done)} completed)")


if __name__ == "__main__":
    main()
