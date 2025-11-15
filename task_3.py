#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Setup & imports
import os, sys, json, math, random, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# from facenet_pytorch import MTCNN, InceptionResnetV1


# In[2]:


# Cell 2: Reproducibility & device
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# In[3]:


# Cell 3 (revised): Paths + read CSV mapping (filename -> label)
DATA_ROOT = Path("/home/akif/Desktop/AS10/archive(1)/Faces/Faces")
CSV_PATH  = Path("/home/akif/Desktop/AS10/archive(1)/Dataset.csv")

assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
assert CSV_PATH.exists(), f"CSV_PATH not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
df["id"] = df["id"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()
df["path"] = df["id"].apply(lambda x: str(DATA_ROOT / x))
df["exists"] = df["path"].apply(lambda p: Path(p).is_file())

missing = df[~df["exists"]]
if len(missing):
    print(f"Warning: {len(missing)} missing files.")
df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

print("Total images:", len(df))
print("Unique identities:", df["label"].nunique())
df.head()


# In[4]:


# Cell 4 (revised): Build {label -> [image paths]} from CSV
from collections import defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
id2imgs = defaultdict(list)
for _, row in df.iterrows():
    p = Path(row["path"])
    if p.suffix.lower() in IMG_EXT:
        id2imgs[row["label"]].append(p)

MIN_PER_ID = 5
id2imgs = {k: v for k, v in id2imgs.items() if len(v) >= MIN_PER_ID}

print("Identities kept (>=5 imgs):", len(id2imgs))
print("Example counts:", {k: len(v) for k, v in list(id2imgs.items())[:5]})


# In[5]:


# Cell 5 (revised): Split identities into employees and outsiders
import random
from math import ceil

all_ids = sorted(id2imgs.keys())
set_seed(42)
random.shuffle(all_ids)

num_employees = min(20, max(5, len(all_ids)//5))
employee_ids = sorted(all_ids[:num_employees])
outsider_ids = sorted(all_ids[num_employees:])

print("Employees selected:", len(employee_ids))
print("Outsiders selected:", len(outsider_ids))
print("Sample employees:", employee_ids[:5])


# In[6]:


# ✅ Cell 6 (OpenCV-based fallback detector + alignment)
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_align(img_path: Path):
    """
    Detects largest face using Haar cascade.
    Returns cropped & resized RGB image (112×112) normalized to [0,1].
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
    face_crop = img[y:y+h, x:x+w]
    if face_crop.size == 0:
        return None
    face_resized = cv2.resize(face_crop, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return face_rgb


# In[7]:


# ✅ Cell 7 (Torch-based face embedding model, no facenet, no insightface)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use pretrained ResNet18 backbone as embedding extractor
class FaceEmbedder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=-1)

# Instantiate
face_model = FaceEmbedder(out_dim=512).to(DEVICE)
face_model.eval()

# Image preprocessing
embed_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

@torch.no_grad()
def embed_batch(faces):
    """
    faces: list of aligned RGB images in [0,1], shape (112,112,3)
    returns: numpy array [N,512]
    """
    if len(faces) == 0:
        return np.zeros((0, 512), dtype=np.float32)
    batch = torch.stack([embed_tfm(Image.fromarray((f*255).astype(np.uint8))) for f in faces]).to(DEVICE)
    embs = face_model(batch)
    return embs.cpu().numpy()


# In[8]:


# Cell 8: Build gallery (and hold-out queries per employee)
# Strategy: for each employee, use N_gallery images to build the gallery, and M_query images for evaluation
N_GALLERY = 8
M_QUERY   = 4  # held-out queries per employee (if available)

gallery = {}          # employee_id -> dict(embeddings: [E_i], centroid: [512], images_gallery: [paths], images_query: [paths])
skipped_ids = []

for emp in tqdm(employee_ids, desc="Building gallery"):
    imgs = id2imgs[emp]
    if len(imgs) < (N_GALLERY + M_QUERY):
        # still allow, but reduce queries
        pass
    # deterministic split
    imgs = sorted(imgs)
    g_paths = imgs[:min(N_GALLERY, len(imgs)//2)]
    q_paths = imgs[min(N_GALLERY, len(imgs)//2): min(len(imgs), min(N_GALLERY, len(imgs)//2) + M_QUERY)]

    # detect+align gallery faces
    g_faces = [detect_align(p) for p in g_paths]
    g_faces = [f for f in g_faces if f is not None]
    if len(g_faces) == 0:
        skipped_ids.append(emp)
        continue
    g_embs = embed_batch(g_faces)
    centroid = g_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    gallery[emp] = {
        "embeddings": g_embs,
        "centroid": centroid,
        "images_gallery": [str(p) for p in g_paths],
        "images_query": [str(p) for p in q_paths]
    }

for emp in skipped_ids:
    employee_ids.remove(emp)

len(gallery), list(gallery.keys())[:5]


# In[9]:


# ✅ Cell 9 (replacement): Rebuild gallery (with a clean split) + create validation embeddings

from tqdm import tqdm
import numpy as np

# You can tweak these
N_GALLERY = 8     # images per employee to form the centroid
M_QUERY   = 4     # held-out query images per employee (genuine)
N_OUT_IDS = 100   # outsiders to sample for impostor validation
MAX_OUT_IMGS_PER_ID = 2

gallery = {}
emp_q_emb_list = []   # genuine queries (employee)
out_emb_list   = []   # impostor queries (outsider)

failed_align = 0

# --- Build gallery + employee query embeddings ---
for e in tqdm(employee_ids, desc="Building gallery + employee queries"):
    paths = id2imgs.get(e, [])
    if not paths:
        continue
    # deterministic order
    paths = sorted(paths)

    # Align all images first
    aligned = []
    for p in paths:
        a = detect_align(p)
        if a is not None:
            aligned.append(a)
        else:
            failed_align += 1

    if len(aligned) < 2:
        # not enough faces for centroid + at least one query
        continue

    # Split into gallery and queries
    g_faces = aligned[:min(N_GALLERY, len(aligned)-1)]
    q_faces = aligned[len(g_faces): len(g_faces)+M_QUERY]

    # If no queries left, skip this id (we need queries for validation)
    if len(q_faces) == 0:
        continue

    # Embed
    g_embs = embed_batch(g_faces)  # [G, D]
    q_embs = embed_batch(q_faces)  # [Q, D]

    # Compute centroid from gallery embeddings
    centroid = g_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Save gallery info
    gallery[e] = {
        "embs": g_embs,
        "centroid": centroid,
        "num_gallery_imgs": len(g_embs),
        "num_query_imgs": len(q_embs)
    }

    # Collect employee query embeddings for validation (genuine)
    if len(q_embs):
        emp_q_emb_list.append(q_embs)

# Stack employee genuine queries
emp_q_embs = np.concatenate(emp_q_emb_list, axis=0) if len(emp_q_emb_list) else np.zeros((0, 512), dtype=np.float32)

# --- Build outsider (impostor) embeddings ---
outsider_pool = outsider_ids[:N_OUT_IDS] if len(outsider_ids) > 0 else []
for oid in tqdm(outsider_pool, desc="Embedding outsiders"):
    paths = sorted(id2imgs.get(oid, []))[:MAX_OUT_IMGS_PER_ID]
    faces = []
    for p in paths:
        a = detect_align(p)
        if a is not None:
            faces.append(a)
        else:
            failed_align += 1
    if faces:
        z = embed_batch(faces)
        if len(z):
            out_emb_list.append(z)

out_embs = np.concatenate(out_emb_list, axis=0) if len(out_emb_list) else np.zeros((0, 512), dtype=np.float32)

print(f"✅ Gallery employees with centroids: {len(gallery)}")
print(f"✅ Employee genuine queries (emb count): {len(emp_q_embs)}")
print(f"✅ Outsider impostor queries (emb count): {len(out_embs)}")
print(f"⚠️ Align failures: {failed_align}")


# In[10]:


# 9.1 🔍 Verify dataset paths
print("CSV path:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print(df.head())

img_dir = Path("/home/akif/Desktop/AS10/archive(1)/Faces/Faces")
print("Directory exists?", img_dir.exists())
print("Total images:", len(list(img_dir.glob('*.jpg'))))

# Check a few absolute paths
for i in range(5):
    sample = df.iloc[i]
    abs_path = img_dir / sample["id"]
    print(abs_path, "→ exists:", abs_path.exists())


# In[11]:


# 9.2 Check a few absolute paths
for i in range(5):
    sample = df.iloc[i]
    abs_path = img_dir / sample["id"]
    print(abs_path, "→ exists:", abs_path.exists())


# In[12]:


# Cell 10
# Siamese dataset generator: returns (img1, img2, label)
from torch.utils.data import Dataset
import random
from PIL import Image

class SiameseFaces(Dataset):
    def __init__(self, id2imgs, transform=None, n_pairs=2000):
        self.items = []
        self.transform = transform
        labels = list(id2imgs.keys())
        for _ in range(n_pairs):
            if random.random() < 0.5:
                # positive pair
                emp = random.choice(labels)
                imgs = random.sample(id2imgs[emp], 2)
                self.items.append((imgs[0], imgs[1], 1))
            else:
                # negative pair
                e1, e2 = random.sample(labels, 2)
                self.items.append((random.choice(id2imgs[e1]), random.choice(id2imgs[e2]), 0))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p1, p2, y = self.items[idx]
        i1, i2 = Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")
        if self.transform:
            i1, i2 = self.transform(i1), self.transform(i2)
        return i1, i2, torch.tensor([y], dtype=torch.float32)


# In[13]:


# Cell 11
from torch.utils.data import DataLoader

# Use your embedder backbone
encoder = FaceEmbedder(out_dim=128).to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
margin = 1.0

def contrastive_loss(z1, z2, y):
    d = F.pairwise_distance(z1, z2)
    loss = y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)
    return loss.mean()

# Build dataset + dataloader
train_ds = SiameseFaces(id2imgs, transform=embed_tfm, n_pairs=2000)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

from tqdm.notebook import tqdm

for epoch in range(5):  # small fine-tune
    encoder.train()
    losses = []
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False)
    for x1, x2, y in pbar:
        x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
        z1, z2 = encoder(x1), encoder(x2)
        loss = contrastive_loss(z1, z2, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": np.mean(losses[-20:])})
    print(f"Epoch {epoch+1} | Avg Loss: {np.mean(losses):.4f}")

# Save fine-tuned encoder
torch.save(encoder.state_dict(), "outputs/siamese_encoder.pt")


# In[14]:


# Cell 12

fine_tuned_encoder = FaceEmbedder(out_dim=128).to(DEVICE)
fine_tuned_encoder.load_state_dict(torch.load("outputs/siamese_encoder.pt"))
fine_tuned_encoder.eval()

@torch.no_grad()
def embed_batch_finetuned(faces):
    if len(faces) == 0:
        return np.zeros((0, 128))
    batch = torch.stack([embed_tfm(Image.fromarray((f*255).astype(np.uint8))) for f in faces]).to(DEVICE)
    z = fine_tuned_encoder(batch)
    return z.cpu().numpy()


# ## 🔧 Siamese Fine-Tuning (Contrastive Loss)
# We fine-tune the ResNet embedding model on same/different face pairs to obtain a
# true **Siamese Network** representation.  
# This improves discrimination before verification and Top-5 ranking.

# In[15]:


# Cell 13
valid_centroids = []
valid_employee_ids = []

for e in employee_ids:
    if e in gallery and "centroid" in gallery[e]:
        valid_centroids.append(gallery[e]["centroid"])
        valid_employee_ids.append(e)
    else:
        print(f"⚠️ Skipping {e}: not found or no centroid")

if len(valid_centroids) == 0:
    raise RuntimeError("❌ No valid centroids found — check gallery creation step above.")

centroids = np.stack(valid_centroids, axis=0)  # [E, 512]

# --- Step 2: Define similarity helper ---
def best_centroid_similarity(q_emb):  # q_emb: [N,512]
    """
    Returns maximum cosine similarity vs any employee centroid.
    """
    if len(q_emb) == 0:
        return np.array([])
    sims = q_emb @ centroids.T  # [N, E]
    return sims.max(axis=1)

# --- Step 3: Compute validation scores ---
val_scores_genuine = best_centroid_similarity(emp_q_embs) if len(emp_q_embs) else np.array([])
val_scores_impostor = best_centroid_similarity(out_embs) if len(out_embs) else np.array([])

# --- Step 4: Build y_true and y_scores vectors ---
y_true = np.concatenate([
    np.ones_like(val_scores_genuine),
    np.zeros_like(val_scores_impostor)
]) if (len(val_scores_genuine) or len(val_scores_impostor)) else np.array([])

y_scores = np.concatenate([
    val_scores_genuine,
    val_scores_impostor
]) if (len(val_scores_genuine) or len(val_scores_impostor)) else np.array([])

# --- Step 5: Quick sanity summary ---
print(f"✅ Valid centroids: {len(valid_centroids)} employees")
print(f"✅ Validation samples: {len(y_true)}")
if len(y_scores):
    print(f"Score range: {y_scores.min():.4f} → {y_scores.max():.4f}")
else:
    print("⚠️ No validation embeddings available.")

(len(y_true), len(y_scores))


# In[16]:


# Cell 14: Threshold selection (EER + option for FMR target)
fpr, tpr, thr = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# EER (Equal Error Rate) threshold
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
EER = (fpr[eer_idx] + fnr[eer_idx]) / 2
T_EER = thr[eer_idx]

# Target false match rate (e.g., 1%)
target_fmr = 0.01
# find threshold where FPR <= target_fmr with highest TPR
valid = np.where(fpr <= target_fmr)[0]
if len(valid) > 0:
    idx = valid[np.argmax(tpr[valid])]
    T_FMR = thr[idx]
    chosen_T = T_FMR
    chosen_note = f"FMR<=1% (TPR={tpr[idx]:.3f}, FPR={fpr[idx]:.3f})"
else:
    chosen_T = T_EER
    chosen_note = f"EER≈{EER:.3f}"

print(f"ROC AUC: {roc_auc:.4f}")
print(f"EER: {EER:.4f}, T_EER: {T_EER:.4f}")
print(f"Chosen threshold: {chosen_T:.4f} ({chosen_note})")

plt.figure(figsize=(4.5,4))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Open-Set ROC")
plt.legend(); plt.grid(alpha=0.3); plt.show()


# In[17]:


# Cell 15: Calibrate similarity -> probability (Platt scaling)
X = y_scores.reshape(-1,1)
scaler = StandardScaler().fit(X)
Xn = scaler.transform(X)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xn, y_true)

def calibrate_prob(sim_array):
    xn = scaler.transform(sim_array.reshape(-1,1))
    p = clf.predict_proba(xn)[:,1]
    return p


# In[18]:


# Cell 16: Build centroid matrix and lookup helpers
emp_ids = employee_ids
centroids = np.stack([gallery[e]["centroid"] for e in emp_ids], axis=0)  # [E,512]

def topk_employees(query_emb: np.ndarray, k=5):
    # query_emb: [512]
    sims = centroids @ (query_emb / (np.linalg.norm(query_emb)+1e-12))
    order = np.argsort(-sims)[:k]
    return [(emp_ids[i], float(sims[i])) for i in order]


# In[19]:


# Cell 17: End-to-end verifier function
def verify_face(image_path: str, threshold: float = None, k: int = 5):
    if threshold is None:
        threshold = chosen_T

    face = detect_align(Path(image_path))
    if face is None:
        return {"ok": False, "reason": "No face detected"}

    emb = embed_batch([face])[0]  # [512]

    # Open-set decision (best centroid similarity)
    sims = centroids @ emb
    best_idx = int(np.argmax(sims))
    best_emp = emp_ids[best_idx]
    best_sim = float(sims[best_idx])

    if best_sim < threshold:
        # Unknown
        return {
            "ok": True,
            "known": False,
            "best_similarity": best_sim,
            "threshold": threshold,
            "message": "Unknown (not a registered employee)"
        }

    # Known -> Closed-set ID
    top = topk_employees(emb, k=k)
    # Calibrated probabilities
    probs = calibrate_prob(np.array([s for _, s in top]))
    top_with_p = [(emp, sim, float(p)) for (emp, sim), p in zip(top, probs)]

    return {
        "ok": True,
        "known": True,
        "predicted_employee": best_emp,
        "best_similarity": best_sim,
        "threshold": threshold,
        "top_k": top_with_p
    }


# In[20]:


# ✅ Cell 18 (updated): Closed-set metrics for known employees
from tqdm import tqdm
import numpy as np

top1_correct = 0
top5_correct = 0
pred_ranks = []

# Use the same key list used when building gallery
emp_ids = list(gallery.keys())

for emp in tqdm(emp_ids, desc="Closed-set eval"):
    # Each employee already has some query embeddings (not paths)
    q_embs = gallery[emp].get("embs_query", None)

    # If you didn’t store query embeddings, rebuild them now:
    if q_embs is None:
        # Take a few images again as queries
        q_paths = id2imgs[emp][:3]  # you can change number
        aligned = [detect_align(p) for p in q_paths if detect_align(p) is not None]
        if not aligned:
            continue
        q_embs = embed_batch(aligned)

    for z in q_embs:
        # Compute cosine similarity to all centroids
        sims = z @ np.stack([gallery[e]["centroid"] for e in emp_ids], axis=0).T
        top_idx = np.argsort(-sims)[:5]
        top_emps = [emp_ids[i] for i in top_idx]

        rank = 6  # default if not in top-5
        for r, e in enumerate(top_emps, start=1):
            if e == emp:
                rank = r
                break
        pred_ranks.append(rank)
        if rank == 1: top1_correct += 1
        if rank <= 5: top5_correct += 1

total = len(pred_ranks)
cmc = np.zeros(5, dtype=np.float32)
for rank in pred_ranks:
    for k in range(5):
        if rank <= (k+1):
            cmc[k] += 1
cmc /= max(1, total)

print(f"Closed-set Top-1: {top1_correct/max(1,total):.3f} | Top-5: {top5_correct/max(1,total):.3f}")
print("CMC (K=1..5):", [round(float(x),3) for x in cmc])


# In[21]:


# Cell 19: Open-set accuracy at chosen threshold
# Known queries = emp_q_embs; Unknown queries = out_embs
def decision_from_score(s, T):
    return 1 if s >= T else 0  # 1=Known, 0=Unknown

known_scores = best_centroid_similarity(emp_q_embs) if len(emp_q_embs) else np.array([])
unk_scores   = best_centroid_similarity(out_embs) if len(out_embs) else np.array([])

known_dec = np.array([decision_from_score(s, chosen_T) for s in known_scores])
unk_dec   = np.array([decision_from_score(s, chosen_T) for s in unk_scores])

known_acc = known_dec.mean() if len(known_dec) else float("nan")
unk_rej   = (1 - unk_dec).mean() if len(unk_dec) else float("nan")

print(f"Open-set @T={chosen_T:.4f} | Known accept rate: {known_acc:.3f} | Unknown rejection rate: {unk_rej:.3f}")


# In[22]:


# ✅ Cell 20 (fixed): Demo — show predictions for a few images with safe JSON printing
import random, json
import numpy as np
from pathlib import Path

def to_py(obj):
    """Recursively convert numpy/scalar/Path types to JSON-safe Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj  # bool, int, float, str, None are already safe

def verify_face(img_path, threshold=0.5, k=5):
    """
    Given an image path, detect + embed face, and find top-k most similar employees.
    Returns ranked list with cosine similarity scores and a 'verified' flag.
    """
    face = detect_align(Path(img_path))
    if face is None:
        return {"error": "No face detected", "path": str(img_path)}

    z = embed_batch([face])[0]  # [512]

    emp_ids_local = list(gallery.keys())
    if not emp_ids_local:
        return {"error": "Gallery is empty — build gallery first.", "path": str(img_path)}

    centroids_mat = np.stack([gallery[e]["centroid"] for e in emp_ids_local], axis=0)  # [E,512]
    sims = z @ centroids_mat.T
    top_idx = np.argsort(-sims)[:k]

    results = [
        {"employee": emp_ids_local[i], "similarity": float(sims[i])}
        for i in top_idx
    ]
    # Force native Python bool
    verified = bool(results[0]["similarity"] >= float(threshold))

    return {
        "path": str(img_path),
        "verified": verified,
        "top_matches": results
    }

# --- Select 3 random query images from your dataset ---
SAMPLES = []
for emp in random.sample(employee_ids, min(3, len(employee_ids))):
    img_list = id2imgs.get(emp, [])
    if len(img_list):
        SAMPLES.append(random.choice(img_list))

print(f"🔍 Showing predictions for {len(SAMPLES)} random faces...\n")

# --- Run verification on chosen samples ---
thr = chosen_T if 'chosen_T' in globals() else 0.5
for img_path in SAMPLES:
    out = verify_face(img_path, threshold=thr, k=5)
    print("\nImage:", img_path)
    print(json.dumps(to_py(out), indent=2))


# In[23]:


# ✅ Cell 21: Save artifacts (centroids, employee list, optional calibrator)
import json, joblib

OUT_DIR = Path("outputs/face_verifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Centroids + ID list
np.save(OUT_DIR / "centroids.npy", np.stack([gallery[e]["centroid"] for e in gallery]))
with open(OUT_DIR / "employee_ids.json", "w") as f:
    json.dump(list(gallery.keys()), f, indent=2)

# Optional: save calibrator and scaler if they exist
if "calibrator" in globals():
    joblib.dump(calibrator, OUT_DIR / "calibrator.pkl")
if "scaler" in globals():
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")

print(f"✅ Saved verifier artifacts in: {OUT_DIR.resolve()}")

