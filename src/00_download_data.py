"""Stage 0 — fetch the BraTS-Path 2026 data from Synapse.

Requires a Synapse account with the challenge data access terms accepted:

    export SYNAPSE_AUTH_TOKEN=...
    python src/00_download_data.py --root ./brats2026

Downloads 40 training shards (~3.9 GB each), the validation shard, the
patch/patient/slide mapping CSV and class_map.json. Already-present files are
skipped, so the script is safe to re-run after an interruption.
"""
import argparse, os

TRAIN_SHARDS = {
    "shard-000000": "syn74904533", "shard-000001": "syn74904532",
    "shard-000002": "syn74904557", "shard-000003": "syn74904556",
    "shard-000004": "syn74904582", "shard-000005": "syn74904607",
    "shard-000006": "syn74904624", "shard-000007": "syn74904637",
    "shard-000008": "syn74904642", "shard-000009": "syn74904659",
    "shard-000010": "syn74904672", "shard-000011": "syn74904714",
    "shard-000012": "syn74904715", "shard-000013": "syn74904750",
    "shard-000014": "syn74904751", "shard-000015": "syn74904760",
    "shard-000016": "syn74904761", "shard-000017": "syn74904803",
    "shard-000018": "syn74904804", "shard-000019": "syn74904824",
    "shard-000020": "syn74904823", "shard-000021": "syn74904848",
    "shard-000022": "syn74904847", "shard-000023": "syn74904868",
    "shard-000024": "syn74904867", "shard-000025": "syn74904893",
    "shard-000026": "syn74904894", "shard-000027": "syn74904907",
    "shard-000028": "syn74904916", "shard-000029": "syn74904939",
    "shard-000030": "syn74904944", "shard-000031": "syn74904979",
    "shard-000032": "syn74904980", "shard-000033": "syn75005013",
    "shard-000034": "syn75005014", "shard-000035": "syn75005060",
    "shard-000036": "syn75005059", "shard-000037": "syn75005099",
    "shard-000038": "syn75005100", "shard-000039": "syn75005113",
}

VAL_SHARD      = "syn75094672"
VAL_CLASS_MAP  = "syn75094354"
TRAIN_CLASS_MAP = "syn74906890"
MAPPING_CSV    = "syn75132830"   # BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv


def fetch(syn, syn_id, dest, label):
    os.makedirs(dest, exist_ok=True)
    print(f"  {label} ({syn_id}) ...", flush=True)
    syn.get(syn_id, downloadLocation=dest)


if __name__ == "__main__":
    import synapseclient

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    ap.add_argument("--skip-train", action="store_true",
                    help="fetch only the validation shard and metadata")
    a = ap.parse_args()

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SystemExit("set SYNAPSE_AUTH_TOKEN (Synapse > Settings > Personal Access Token)")

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    train = f"{a.root}/data/train/"
    val   = f"{a.root}/data/val/"

    print("metadata:")
    fetch(syn, MAPPING_CSV, train, "patch/patient/slide mapping CSV")
    fetch(syn, TRAIN_CLASS_MAP, train, "train class_map.json")
    fetch(syn, VAL_CLASS_MAP, val, "val class_map.json")

    if not a.skip_train:
        have = set(os.listdir(train)) if os.path.isdir(train) else set()
        todo = {n: s for n, s in TRAIN_SHARDS.items() if f"{n}.tar" not in have}
        print(f"\ntraining shards: {len(TRAIN_SHARDS) - len(todo)} present, {len(todo)} to fetch")
        for name, syn_id in todo.items():
            fetch(syn, syn_id, train, name)

    print("\nvalidation shard:")
    if not os.path.exists(os.path.join(val, "val-shard-000000.tar")):
        fetch(syn, VAL_SHARD, val, "val-shard-000000")
    else:
        print("  already present")

    print("\ndone")
