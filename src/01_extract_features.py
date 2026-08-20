"""Stage 1 — extract and cache patch embeddings from both frozen backbones.

H-optimus-1  -> 1536-d pooled embedding
Virchow2     -> 2560-d  [CLS ; mean(patch tokens)]

Neither encoder is trained. Every patch is embedded once and cached as .npz,
so all later stages operate on stored feature vectors.

    export HF_TOKEN=...
    python src/01_extract_features.py --backbone hoptimus --split train
    python src/01_extract_features.py --backbone virchow2 --split train
    python src/01_extract_features.py --backbone hoptimus --split val
    python src/01_extract_features.py --backbone virchow2 --split val
    python src/01_extract_features.py --labels            # writes train_labels_full.npz
"""
import argparse, glob, os
import numpy as np, torch, timm, webdataset as wds
from huggingface_hub import login
from torchvision import transforms
from tqdm import tqdm

CLASSES = ["CT", "DM", "IC", "LI", "MP", "NC", "PL", "PN", "WM", "NOTA"]

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD  = (0.211883, 0.230117, 0.177517)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build(backbone, device):
    """Return (model, transform, embedding_fn) for one frozen backbone."""
    if backbone == "hoptimus":
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=False)
        mean, std = HOPTIMUS_MEAN, HOPTIMUS_STD
        embed = lambda out: out                       # already pooled, 1536-d
    elif backbone == "virchow2":
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                  mlp_layer=timm.layers.SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        # class token concatenated with the mean of the spatial patch tokens
        embed = lambda out: torch.cat([out[:, 0], out[:, 1:].mean(dim=1)], dim=1)
    else:
        raise ValueError(backbone)

    tf = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=mean, std=std)])
    return model.to(device).eval(), tf, embed


def extract(shards, out_dir, model, tf, embed, device, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    done = {f[:-4] for f in os.listdir(out_dir) if f.endswith(".npz")}

    for path in shards:
        name = os.path.basename(path).replace(".tar", "")
        if name in done:
            print(f"skip {name} (already extracted)"); continue

        ds = (wds.WebDataset(path, shardshuffle=False)
                .decode("pil").to_tuple("__key__", "jpg")
                .map_tuple(lambda x: x, tf))

        embs, names, bimg, bname = [], [], [], []

        def flush():
            if not bimg: return
            with torch.no_grad(), torch.amp.autocast("cuda"):
                out = model(torch.stack(bimg).to(device))
                embs.append(embed(out).cpu().float().numpy())
            names.extend(bname); bimg.clear(); bname.clear()

        for fn, img in tqdm(ds, desc=name):
            bimg.append(img); bname.append(fn)
            if len(bimg) == batch_size: flush()
        flush()

        E = np.concatenate(embs, 0)
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                            embeddings=E, filenames=np.array(names))
        print(f"saved {name}.npz  {E.shape}")
        torch.cuda.empty_cache()


def extract_labels(shards, out_path):
    """Read the .cls entry of every training sample."""
    names, labels = [], []
    to_int = lambda x: int(x.decode() if isinstance(x, bytes) else x)
    for path in shards:
        ds = (wds.WebDataset(path, shardshuffle=False)
                .decode("pil").to_tuple("__key__", "cls")
                .map_tuple(lambda x: x, to_int))
        for fn, lab in tqdm(ds, desc=os.path.basename(path)):
            names.append(fn); labels.append(lab)
    np.savez_compressed(out_path, labels=np.array(labels),
                        filenames=np.array(names))
    print(f"saved {out_path}  n={len(labels):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    ap.add_argument("--backbone", choices=["hoptimus", "virchow2"])
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    pattern = "shard-*.tar" if a.split == "train" else "val-shard-*.tar"
    shards = sorted(glob.glob(f"{a.root}/data/{a.split}/{pattern}"))
    if not shards:
        raise SystemExit(f"no shards under {a.root}/data/{a.split}/")

    if a.labels:
        extract_labels(shards, f"{a.root}/embeddings/train_labels_full.npz")
        raise SystemExit(0)

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, embed = build(a.backbone, device)
    suffix = "" if a.backbone == "hoptimus" else "_virchow2"
    extract(shards, f"{a.root}/embeddings/{a.split}{suffix}/",
            model, tf, embed, device, a.batch_size)
