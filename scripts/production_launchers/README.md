# Production section launchers

Copy these files into your PyCIL repository:

- `production_section.py` -> `scripts/production_section.py`
- `run_production_pc2.ps1` -> `scripts/run_production_pc2.ps1`
- `run_production_server.sh` -> `scripts/run_production_server.sh`

## PC2 / RTX 3070 sections

1. FeCAM AID T5 R18, batch 20
2. FeCAM AID T10 R18, batch 20
3. SSRE AID T5 R18, batch 20
4. SSRE AID T10 R18, batch 20

Each section runs:
`none -> eca -> cbam -> procbam`.

Run:
```powershell
.\scripts\run_production_pc2.ps1 -Section 1
```

## Server / RTX A5000 sections

1. FeTrIL AID T5 R18, batch 32
2. FeTrIL AID T10 R18, batch 32
3. FeTrIL AID T5 R34, batch 32
4. FeTrIL AID T10 R34, batch 32
5. EFC AID T5 R18, batch 12
6. EFC AID T10 R18, batch 12
7. EFC AID T5 R34, batch 12
8. EFC AID T10 R34, batch 12
9. FeCAM AID T5 R34, batch 64
10. FeCAM AID T10 R34, batch 64
11. SSRE AID T5 R34, batch 32
12. SSRE AID T10 R34, batch 32

Each section runs:
`none -> eca -> cbam -> procbam`.

Run:
```bash
bash scripts/run_production_server.sh 1
```

PASS is intentionally excluded from the active production queue.

The Python orchestrator uses `run_mode="experiment"`, so checkpoints are ON.
It validates every completed run and writes a section report.
Completed runs are skipped by the runner's existing skip-completed behavior.
